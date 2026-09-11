# Audit chaincode and the Fabric test-network

`auditcontract/` is the Go chaincode of Section IV-E: `LogAccess`, `QueryByPrincipal`,
`QueryByResource`, `VerifyChain`, plus a one-time `SetPDPPublicKey`. There is no update or delete
transaction. `LogAccess` verifies the PDP's ECDSA-P-256 signature over the canonical record
(`canonical.go` is a byte-exact port of `ztb/pdp/signer.py`; the Go tests verify Python-made
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

```bash
cd ~/fabric-samples/test-network
./network.sh down
./network.sh up createChannel -c mychannel -ca -s couchdb
./network.sh deployCC -c mychannel -ccn auditcontract \
    -ccp /Users/harshgupta/projects/ztbaudit/chaincode/auditcontract -ccl go \
    -ccep "OR('Org1MSP.peer','Org2MSP.peer')"
```

`-s couchdb` is required for `QueryByResource` (a rich query; the index ships in
`META-INF/statedb/couchdb/indexes/`). `QueryByPrincipal` and `VerifyChain` use a composite key
and work on LevelDB too. The endorsement policy above is for the demo; the paper's testbed uses
`OutOf(2, ...)` across three organisations (see below).

## Gateway shim and the PDP

```bash
cd ztb/ledger/shim && npm install && node server.js      # :7071, Org1 User1 identity
ZTB_LEDGER=fabric make serve                              # PDP commits through the shim
curl -s localhost:8000/verify/CDE1846
```

On first use the PDP installs its public key with `SetPDPPublicKey`; the key can be set only
once per channel, so a new PDP key means a fresh channel (or redeploying the chaincode).

## Experiments

```bash
python scripts/tamper_test.py --ledger fabric --out results      # Table VII
python scripts/ledger_bench.py --ledger fabric --out results     # Fig. 6
```

Without Docker, `--ledger sim` runs both against `ztb/ledger/sim.py`, a Python reference
implementation of exactly the chaincode's rules; its Table VII is labelled `ledger=sim` and its
throughput must not be reported as ledger throughput.

Against a real Fabric network the tamper experiment's "direct state edit" is modelled as
suppression plus resubmission: the world state is writable only through endorsed transactions,
so an administrator would have to edit CouchDB on every endorsing peer; edited records are
resubmitted through `LogAccess` and rejected, deletions are detected by `VerifyChain`.

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

## Residual risk

`VerifyChain` reads the world state. An adversary who deletes a principal's tail records **and**
rewrites that principal's head pointer consistently defeats it at the world-state level; the block
store still holds the original transactions, so the ledger-level check of Section IV-F (re-validating
the block chain and endorsements) or periodic Merkle-root anchoring closes that gap.
