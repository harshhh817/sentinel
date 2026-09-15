# Audit chaincode and the Fabric test-network

`auditcontract/` is the Go chaincode of Section IV-E: `LogAccess`, `QueryByPrincipal`,
`QueryByResource`, `VerifyChain`, plus a one-time `SetPDPPublicKey`. There is no update or delete
transaction. `LogAccess` verifies the PDP's ECDSA-P-256 signature over the canonical record
(`canonical.go` is a byte-exact port of `sentinel/pdp/signer.py`; the Go tests verify Python-made
signatures from `testdata/vectors.json`), enforces per-principal `seq` continuity and `prevHash`,
and rejects duplicate `recId`. `VerifyChain` re-walks a principal's chain and also checks the
record count against the stored head, so a deleted *last* record is caught too.

Unit tests need no network:

```bash
cd chaincode/auditcontract && go test ./...
```

## Prerequisites

Docker Desktop (the test-network runs peers, orderer and CouchDB as containers), Go ≥ 1.22,
Node ≥ 18 (for the gateway shim), and fabric-samples with the Fabric 2.5 binaries:

```bash
cd ~ && curl -sSL https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh \
  | bash -s -- --fabric-version 2.5.9 docker samples binary
```

## Bring up a 2-org network with CouchDB and deploy

The chaincode is deployed **as a service** (CCaaS): the image is built on the host from the
`Dockerfile` (vendored modules, `go mod vendor`) and each peer connects to a running container.
Fabric 2.5's classic `deployCC` path builds the image *inside* the peer through the host Docker
socket with a legacy API, and current Docker Desktop engines (29.x) drop that connection with an
empty build log; CCaaS avoids the in-peer build entirely.

```bash
cd ~/fabric-samples/test-network
./network.sh down
./network.sh up createChannel -c mychannel -ca -s couchdb
docker build -t auditcontract_ccaas_image:latest /Users/harshgupta/projects/sentinel/chaincode/auditcontract
./network.sh deployCCAAS -c mychannel -ccn auditcontract \
    -ccp /Users/harshgupta/projects/sentinel/chaincode/auditcontract \
    -ccep "OR('Org1MSP.peer','Org2MSP.peer')"          # add -ccs N to redeploy a new image
```

`main()` starts a `shim.ChaincodeServer` when `CHAINCODE_SERVER_ADDRESS` is set (deployCCAAS
passes it together with `CHAINCODE_ID`). Note for anyone extending the record: `contractapi`
rejects pointer fields in returned structs, which is why the nullable `r`/`R` are `any`.

`-s couchdb` is required for `QueryByResource` (a rich query; the index ships in
`META-INF/statedb/couchdb/indexes/`). `QueryByPrincipal` and `VerifyChain` use a composite key
and work on LevelDB too. The endorsement policy above is for the demo; the paper's testbed uses
`OutOf(2, ...)` across three organisations (see below).

## Gateway shim and the PDP

```bash
cd sentinel/ledger/shim && npm install && node server.js      # :7071, Org1 User1 identity
SENTINEL_LEDGER=fabric make serve                              # PDP commits through the shim
curl -s localhost:8000/verify/CDE1846
```

On first use the PDP installs its public key with `SetPDPPublicKey`; the key can be set only
once per channel, so a new PDP key means a fresh channel (or redeploying the chaincode).

## Experiments

```bash
python scripts/tamper_test.py  --ledger fabric --keys-dir state/keys \
    --couchdb http://admin:adminpw@localhost:5984/mychannel_auditcontract --out results   # Table VII
python scripts/ledger_bench.py --ledger fabric --keys-dir state/keys --out results         # Fig. 6
```

`--keys-dir` persists the PDP key so both scripts share one channel (the chaincode accepts the
key once). `--couchdb` is how the tamper experiment models A3 on a live network: it edits and
deletes the endorsing peer's CouchDB documents directly, behind the chaincode's back, so
`VerifyChain` has to catch it from the state alone. Commits are parallelised across principals
(each chain stays in order) because every Gateway submit waits ~2 s for block commit.

Without Docker, `--ledger sim` runs both against `sentinel/ledger/sim.py`, a Python reference
implementation of exactly the chaincode's rules; its Table VII is labelled `ledger=sim` and its
throughput must not be reported as ledger throughput.

Fabricated records go through `LogAccess` and are rejected at endorsement; deletions,
verdict edits and backdating are applied straight to the peer's CouchDB and are detected by
`VerifyChain` (a gap, or a signature that no longer verifies).

## Extending to three organisations (the paper's Table IV)

test-network ships a third-org add-on:

```bash
cd ~/fabric-samples/test-network/addOrg3
./addOrg3.sh up -c mychannel -ca -s couchdb
cd .. && ./network.sh deployCC -c mychannel -ccn auditcontract -ccp <path> -ccl go \
    -ccep "OutOf(2,'Org1MSP.peer','Org2MSP.peer','Org3MSP.peer')"
```

That gives three organisations with one peer each (the paper has two peers per org and a
three-node Raft orderer; `test-network` uses a single orderer — for the demo the endorsement
policy is what matters). Point additional shim instances at `peer0.org2` / `peer0.org3` by
setting `PEER`, `PEER_HOST_ALIAS`, `MSP_ID` and the org's crypto path.

## Peer state cache

The peer keeps an in-memory cache in front of CouchDB. A direct CouchDB edit is therefore
invisible to `VerifyChain` on that peer until the key is evicted or the peer restarts — the
tamper experiment only saw its edits because 10,000 control writes had churned the cache. For
the demo the peers run with `CORE_LEDGER_STATE_COUCHDBCONFIG_CACHESIZE=0` (added to
`compose/compose-couch.yaml`; recreate the peers with `docker compose ... up -d --no-deps
peer0.org1.example.com peer0.org2.example.com` under the `compose` project name so the ledger
volumes reattach). In a real deployment the auditor verifies on a peer they control, whose cache
the adversary never touched — the same reason the other organisation's peer shows the chain
intact while the tampered organisation's does not.

## Residual risk

`VerifyChain` reads the world state. An adversary who deletes a principal's tail records **and**
rewrites that principal's head pointer consistently defeats it at the world-state level; the block
store still holds the original transactions, so the ledger-level check of Section IV-F (re-validating
the block chain and endorsements) or periodic Merkle-root anchoring closes that gap.
