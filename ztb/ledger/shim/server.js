// HTTP/JSON front for the audit contract, using the official Fabric Gateway client.
// Env (defaults match fabric-samples/test-network, Org1 peer0):
//   FABRIC_SAMPLES   path to fabric-samples            (default ~/fabric-samples)
//   CHANNEL          mychannel     CHAINCODE  auditcontract
//   MSP_ID           Org1MSP       PEER       localhost:7051
//   PEER_HOST_ALIAS  peer0.org1.example.com   PORT 7071
import { createServer } from 'node:http';
import { promises as fs } from 'node:fs';
import { createPrivateKey, createHash } from 'node:crypto';
import path from 'node:path';
import os from 'node:os';
import * as grpc from '@grpc/grpc-js';
import { connect, signers } from '@hyperledger/fabric-gateway';

const SAMPLES = process.env.FABRIC_SAMPLES ?? path.join(os.homedir(), 'fabric-samples');
const ORG = path.join(SAMPLES, 'test-network/organizations/peerOrganizations/org1.example.com');
const CHANNEL = process.env.CHANNEL ?? 'mychannel';
const CHAINCODE = process.env.CHAINCODE ?? 'auditcontract';
const MSP_ID = process.env.MSP_ID ?? 'Org1MSP';
const PEER = process.env.PEER ?? 'localhost:7051';
const PEER_ALIAS = process.env.PEER_HOST_ALIAS ?? 'peer0.org1.example.com';
const PORT = Number(process.env.PORT ?? 7071);

async function firstFile(dir) { const [f] = await fs.readdir(dir); return path.join(dir, f); }

async function gateway() {
  const tlsCert = await fs.readFile(path.join(ORG, 'peers/peer0.org1.example.com/tls/ca.crt'));
  const client = new grpc.Client(PEER, grpc.credentials.createSsl(tlsCert),
    { 'grpc.ssl_target_name_override': PEER_ALIAS });
  const certPath = await firstFile(path.join(ORG, 'users/User1@org1.example.com/msp/signcerts'));
  const keyPath = await firstFile(path.join(ORG, 'users/User1@org1.example.com/msp/keystore'));
  const identity = { mspId: MSP_ID, credentials: await fs.readFile(certPath) };
  const signer = signers.newPrivateKeySigner(createPrivateKey(await fs.readFile(keyPath)));
  const gw = connect({ client, identity, signer,
    evaluateOptions: () => ({ deadline: Date.now() + 60000 }),
    endorseOptions: () => ({ deadline: Date.now() + 60000 }),
    submitOptions: () => ({ deadline: Date.now() + 60000 }),
    commitStatusOptions: () => ({ deadline: Date.now() + 120000 }) });
  return gw.getNetwork(CHANNEL).getContract(CHAINCODE);
}

const contract = await gateway();
const dec = new TextDecoder();
const parse = (b) => { const s = dec.decode(b); return s ? JSON.parse(s) : null; };

async function body(req) {
  const chunks = []; for await (const c of req) chunks.push(c);
  const s = Buffer.concat(chunks).toString(); return s ? JSON.parse(s) : {};
}

const routes = {
  '/health': async () => ({ status: 'ok', channel: CHANNEL, chaincode: CHAINCODE }),
  '/SetPDPPublicKey': async ({ pem }) => { await contract.submitTransaction('SetPDPPublicKey', pem); return { ok: true }; },
  '/LogAccess': async ({ record }) => { await contract.submitTransaction('LogAccess', JSON.stringify(record)); return { ok: true, recId: record.recId }; },
  '/QueryByPrincipal': async ({ principal }) => parse(await contract.evaluateTransaction('QueryByPrincipal', principal)),
  '/QueryByResource': async ({ resource }) => parse(await contract.evaluateTransaction('QueryByResource', resource)),
  '/VerifyChain': async ({ principal }) => parse(await contract.evaluateTransaction('VerifyChain', principal)),
  '/all': async () => { throw Object.assign(new Error('use QueryByPrincipal / QueryByResource'), { status: 400 }); },
};

createServer(async (req, res) => {
  const route = routes[req.url];
  if (!route) { res.writeHead(404); return res.end(); }
  try {
    const out = await route(req.method === 'POST' ? await body(req) : {});
    res.writeHead(200, { 'content-type': 'application/json' }); res.end(JSON.stringify(out));
  } catch (e) {
    // Endorsement failures carry the chaincode's message; surface them as 422 so the
    // Python client can raise LedgerRejected instead of a transport error.
    const msg = e?.details?.map?.(d => d.message).join('; ') || e.message || String(e);
    const status = e.status ?? (/duplicate|signature|discontinuity|prevHash|required|not set|already set/i.test(msg) ? 422 : 500);
    res.writeHead(status, { 'content-type': 'text/plain' }); res.end(msg);
  }
}).listen(PORT, () => console.log(`ztb fabric shim on :${PORT} -> ${CHANNEL}/${CHAINCODE} via ${PEER}`));
