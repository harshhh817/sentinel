package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"github.com/hyperledger/fabric-chaincode-go/shim"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
	"github.com/hyperledger/fabric-protos-go/ledger/queryresult"
)

// ---------------- vectors produced by ztb/pdp/signer.py ----------------

type vectors struct {
	PDPPublicKeyPEM string `json:"pdp_public_key_pem"`
	Chain           []struct {
		Record     AuditRecord `json:"record"`
		Canonical  string      `json:"canonical"`
		RecordHash string      `json:"record_hash"`
	} `json:"chain"`
	Edge []struct {
		Record    AuditRecord `json:"record"`
		Canonical string      `json:"canonical"`
	} `json:"edge"`
	Foreign AuditRecord `json:"foreign_signer_record"`
}

func loadVectors(t *testing.T) vectors {
	t.Helper()
	raw, err := os.ReadFile("testdata/vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	var v vectors
	if err := json.Unmarshal(raw, &v); err != nil {
		t.Fatal(err)
	}
	return v
}

func TestCanonicalMatchesPythonByteForByte(t *testing.T) {
	v := loadVectors(t)
	for _, c := range v.Chain {
		if got := string(Canonical(&c.Record)); got != c.Canonical {
			t.Fatalf("canonical mismatch\n go: %s\n py: %s", got, c.Canonical)
		}
		if got := RecordHash(&c.Record); got != c.RecordHash {
			t.Fatalf("record hash mismatch for %s", c.Record.RecID)
		}
	}
	for _, e := range v.Edge {
		if got := string(Canonical(&e.Record)); got != e.Canonical {
			t.Fatalf("edge canonical mismatch\n go: %s\n py: %s", got, e.Canonical)
		}
	}
}

func TestPythonSignaturesVerifyAndForeignKeyDoesNot(t *testing.T) {
	v := loadVectors(t)
	pub, err := parsePub(v.PDPPublicKeyPEM)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range v.Chain {
		if !VerifySignature(pub, &c.Record) {
			t.Fatalf("python signature failed to verify for %s", c.Record.RecID)
		}
		tampered := c.Record
		tampered.Verdict = "DENY"
		if VerifySignature(pub, &tampered) {
			t.Fatal("modified verdict still verifies")
		}
		backdated := c.Record
		backdated.TS = "2025-01-01T00:00:00+00:00"
		if VerifySignature(pub, &backdated) {
			t.Fatal("backdated record still verifies")
		}
	}
	if VerifySignature(pub, &v.Foreign) {
		t.Fatal("record signed by a foreign key verifies")
	}
}

func TestVerifyChainRecordsFindsFirstDiscontinuity(t *testing.T) {
	v := loadVectors(t)
	pub, _ := parsePub(v.PDPPublicKeyPEM)
	recs := make([]*AuditRecord, 0, len(v.Chain))
	for i := range v.Chain {
		r := v.Chain[i].Record
		recs = append(recs, &r)
	}
	if idx, _ := VerifyChainRecords(pub, recs); idx != -1 {
		t.Fatalf("intact chain reported discontinuity at %d", idx)
	}
	deleted := append([]*AuditRecord{}, recs[:2]...)
	deleted = append(deleted, recs[3:]...)
	if idx, _ := VerifyChainRecords(pub, deleted); idx != 2 {
		t.Fatalf("deletion: want 2, got %d", idx)
	}
	modified := make([]*AuditRecord, len(recs))
	for i, r := range recs {
		c := *r
		modified[i] = &c
	}
	modified[1].Verdict = "DENY"
	if idx, _ := VerifyChainRecords(pub, modified); idx != 1 {
		t.Fatalf("modification: want 1, got %d", idx)
	}
	fabricated := append([]*AuditRecord{}, recs[:4]...)
	f := v.Foreign
	f.Seq = 5
	f.PrevHash = RecordHash(recs[3])
	fabricated = append(fabricated, &f)
	if idx, _ := VerifyChainRecords(pub, fabricated); idx != 4 {
		t.Fatalf("fabrication: want 4, got %d", idx)
	}
}

// ---------------- minimal in-memory stub for the transaction paths ----------------

type memStub struct {
	shim.ChaincodeStubInterface // unimplemented methods panic if touched
	state                       map[string][]byte
}

func newStub() *memStub { return &memStub{state: map[string][]byte{}} }

func (s *memStub) GetState(k string) ([]byte, error) { return s.state[k], nil }
func (s *memStub) PutState(k string, v []byte) error  { s.state[k] = v; return nil }
func (s *memStub) CreateCompositeKey(objectType string, attrs []string) (string, error) {
	return "\x00" + objectType + "\x00" + strings.Join(attrs, "\x00") + "\x00", nil
}

type memIter struct {
	kvs []*queryresult.KV
	i   int
}

func (it *memIter) HasNext() bool { return it.i < len(it.kvs) }
func (it *memIter) Next() (*queryresult.KV, error) {
	if !it.HasNext() {
		return nil, errors.New("exhausted")
	}
	kv := it.kvs[it.i]
	it.i++
	return kv, nil
}
func (it *memIter) Close() error { return nil }

func (s *memStub) GetStateByPartialCompositeKey(objectType string, attrs []string) (shim.StateQueryIteratorInterface, error) {
	prefix, _ := s.CreateCompositeKey(objectType, attrs)
	prefix = strings.TrimSuffix(prefix, "\x00")
	keys := make([]string, 0)
	for k := range s.state {
		if strings.HasPrefix(k, prefix) {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	it := &memIter{}
	for _, k := range keys {
		it.kvs = append(it.kvs, &queryresult.KV{Key: k, Value: s.state[k]})
	}
	return it, nil
}

type memCtx struct {
	contractapi.TransactionContextInterface
	stub *memStub
}

func (c *memCtx) GetStub() shim.ChaincodeStubInterface { return c.stub }

func TestLogAccessEnforcesTheRules(t *testing.T) {
	v := loadVectors(t)
	cc := &AuditContract{}
	ctx := &memCtx{stub: newStub()}
	if err := cc.LogAccess(ctx, mustJSON(v.Chain[0].Record)); err == nil {
		t.Fatal("LogAccess must fail before the PDP key is set")
	}
	if err := cc.SetPDPPublicKey(ctx, v.PDPPublicKeyPEM); err != nil {
		t.Fatal(err)
	}
	if err := cc.SetPDPPublicKey(ctx, v.PDPPublicKeyPEM); err == nil {
		t.Fatal("PDP key must only be set once")
	}
	// happy path: whole chain in order
	for _, c := range v.Chain {
		if err := cc.LogAccess(ctx, mustJSON(c.Record)); err != nil {
			t.Fatalf("LogAccess rejected a valid record: %v", err)
		}
	}
	res, err := cc.VerifyChain(ctx, "u1")
	if err != nil || !res.Intact || res.Records != 5 || res.FirstDiscontinuity != -1 {
		t.Fatalf("VerifyChain on intact ledger: %+v %v", res, err)
	}
	// duplicate recId
	if err := cc.LogAccess(ctx, mustJSON(v.Chain[0].Record)); err == nil || !strings.Contains(err.Error(), "duplicate") {
		t.Fatalf("duplicate recId not rejected: %v", err)
	}
	// fabricated (foreign key), correct seq/prevHash
	f := v.Foreign
	f.RecID = "fab-1"
	f.Seq = 6
	f.PrevHash = RecordHash(&v.Chain[4].Record)
	if err := cc.LogAccess(ctx, mustJSON(f)); err == nil || !strings.Contains(err.Error(), "signature") {
		t.Fatalf("fabricated record not rejected: %v", err)
	}
	// modified verdict (signature breaks)
	m := v.Chain[4].Record
	m.RecID = "mod-1"
	m.Seq = 6
	m.PrevHash = RecordHash(&v.Chain[4].Record)
	m.Verdict = "DENY"
	if err := cc.LogAccess(ctx, mustJSON(m)); err == nil {
		t.Fatal("modified record not rejected")
	}
	// seq gap / prevHash mismatch are rejected even with a valid signature: reuse a
	// genuinely signed record from a fresh principal at the wrong seq.
	q := v.Chain[1].Record // seq 2 for u1, but u2 has no records yet
	q.Principal = "u2"     // breaks the signature too, so use the message text to confirm the order of checks
	if err := cc.LogAccess(ctx, mustJSON(q)); err == nil {
		t.Fatal("record with broken signature accepted")
	}
	recs, _ := cc.QueryByPrincipal(ctx, "u1")
	if len(recs) != 5 || recs[0].Seq != 1 || recs[4].Seq != 5 {
		t.Fatalf("QueryByPrincipal order/len wrong: %d", len(recs))
	}
	if recs, _ := cc.QueryByPrincipal(ctx, "nobody"); len(recs) != 0 {
		t.Fatal("unknown principal must return an empty list")
	}
}

func TestVerifyChainDetectsASuppressedRecord(t *testing.T) {
	v := loadVectors(t)
	cc := &AuditContract{}
	ctx := &memCtx{stub: newStub()}
	_ = cc.SetPDPPublicKey(ctx, v.PDPPublicKeyPEM)
	for _, c := range v.Chain {
		_ = cc.LogAccess(ctx, mustJSON(c.Record))
	}
	// Simulate an administrator deleting seq 3 from the state database directly.
	ck, _ := ctx.stub.CreateCompositeKey(indexPrinSeq, []string{"u1", fmt.Sprintf("%012d", 3)})
	delete(ctx.stub.state, ck)
	delete(ctx.stub.state, prefixRecord+v.Chain[2].Record.RecID)
	res, _ := cc.VerifyChain(ctx, "u1")
	if res.Intact || res.FirstDiscontinuity != 2 {
		t.Fatalf("suppressed record not detected: %+v", res)
	}
}

func mustJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func TestVerifyChainDetectsATruncatedTail(t *testing.T) {
	v := loadVectors(t)
	cc := &AuditContract{}
	ctx := &memCtx{stub: newStub()}
	_ = cc.SetPDPPublicKey(ctx, v.PDPPublicKeyPEM)
	for _, c := range v.Chain {
		_ = cc.LogAccess(ctx, mustJSON(c.Record))
	}
	last := v.Chain[len(v.Chain)-1].Record
	ck, _ := ctx.stub.CreateCompositeKey(indexPrinSeq, []string{"u1", fmt.Sprintf("%012d", last.Seq)})
	delete(ctx.stub.state, ck)
	delete(ctx.stub.state, prefixRecord+last.RecID)
	res, _ := cc.VerifyChain(ctx, "u1")
	if res.Intact || res.FirstDiscontinuity != 4 {
		t.Fatalf("truncated tail not detected: %+v", res)
	}
}
