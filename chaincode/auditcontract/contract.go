package main

import (
	"crypto/ecdsa"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"strconv"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// AuditContract exposes the four transactions of Section IV-E. There is deliberately
// no update or delete: revision can only take the form of an appended contradiction.
type AuditContract struct {
	contractapi.Contract
}

const (
	keyPDPPub    = "config/pdp_public_key"
	prefixRecord = "rec/"
	prefixHead   = "head/"
	indexPrinSeq = "principal~seq"
)

type chainHead struct {
	Seq      int64  `json:"seq"`
	PrevHash string `json:"prevHash"`
}

// VerifyResult is what VerifyChain returns.
type VerifyResult struct {
	Principal          string `json:"principal"`
	Records            int    `json:"records"`
	Intact             bool   `json:"intact"`
	FirstDiscontinuity int    `json:"firstDiscontinuity"` // -1 when intact
	Reason             string `json:"reason,omitempty"`
}

// SetPDPPublicKey installs the PDP's P-256 public key (PEM). It can be set once;
// rotating it is an organisational decision made by redeploying the chaincode.
func (c *AuditContract) SetPDPPublicKey(ctx contractapi.TransactionContextInterface, pemKey string) error {
	existing, err := ctx.GetStub().GetState(keyPDPPub)
	if err != nil {
		return err
	}
	if existing != nil {
		return fmt.Errorf("PDP public key already set")
	}
	if _, err := parsePub(pemKey); err != nil {
		return err
	}
	return ctx.GetStub().PutState(keyPDPPub, []byte(pemKey))
}

func parsePub(pemKey string) (*ecdsa.PublicKey, error) {
	block, _ := pem.Decode([]byte(pemKey))
	if block == nil {
		return nil, fmt.Errorf("invalid PEM")
	}
	pub, err := x509.ParsePKIXPublicKey(block.Bytes)
	if err != nil {
		return nil, err
	}
	ec, ok := pub.(*ecdsa.PublicKey)
	if !ok {
		return nil, fmt.Errorf("not an ECDSA key")
	}
	return ec, nil
}

func (c *AuditContract) pdpKey(ctx contractapi.TransactionContextInterface) (*ecdsa.PublicKey, error) {
	raw, err := ctx.GetStub().GetState(keyPDPPub)
	if err != nil {
		return nil, err
	}
	if raw == nil {
		return nil, fmt.Errorf("PDP public key not set")
	}
	return parsePub(string(raw))
}

// VerifySignature checks the PDP's ECDSA-P256/SHA-256 DER signature over Canonical(rec).
func VerifySignature(pub *ecdsa.PublicKey, rec *AuditRecord) bool {
	sig, err := base64.StdEncoding.DecodeString(rec.Sig)
	if err != nil || len(sig) == 0 {
		return false
	}
	digest := sha256.Sum256(Canonical(rec))
	return ecdsa.VerifyASN1(pub, digest[:], sig)
}

// LogAccess appends one audit record. It is rejected if the recId already exists,
// the signature does not verify against the PDP key, seq is not head.seq+1, or
// prevHash is not the hash of the principal's previous record.
func (c *AuditContract) LogAccess(ctx contractapi.TransactionContextInterface, recordJSON string) error {
	var rec AuditRecord
	if err := json.Unmarshal([]byte(recordJSON), &rec); err != nil {
		return fmt.Errorf("malformed record: %w", err)
	}
	if rec.RecID == "" || rec.Principal == "" {
		return fmt.Errorf("recId and principal are required")
	}
	stub := ctx.GetStub()
	if existing, err := stub.GetState(prefixRecord + rec.RecID); err != nil {
		return err
	} else if existing != nil {
		return fmt.Errorf("duplicate recId %s", rec.RecID)
	}
	pub, err := c.pdpKey(ctx)
	if err != nil {
		return err
	}
	if !VerifySignature(pub, &rec) {
		return fmt.Errorf("signature does not verify for recId %s", rec.RecID)
	}
	head := chainHead{Seq: 0, PrevHash: Genesis}
	if raw, err := stub.GetState(prefixHead + rec.Principal); err != nil {
		return err
	} else if raw != nil {
		if err := json.Unmarshal(raw, &head); err != nil {
			return err
		}
	}
	if rec.Seq != head.Seq+1 {
		return fmt.Errorf("seq discontinuity for %s: got %d, expected %d", rec.Principal, rec.Seq, head.Seq+1)
	}
	if rec.PrevHash != head.PrevHash {
		return fmt.Errorf("prevHash mismatch for %s at seq %d", rec.Principal, rec.Seq)
	}
	raw, _ := json.Marshal(rec)
	if err := stub.PutState(prefixRecord+rec.RecID, raw); err != nil {
		return err
	}
	ck, err := stub.CreateCompositeKey(indexPrinSeq, []string{rec.Principal, fmt.Sprintf("%012d", rec.Seq)})
	if err != nil {
		return err
	}
	if err := stub.PutState(ck, []byte(rec.RecID)); err != nil {
		return err
	}
	newHead, _ := json.Marshal(chainHead{Seq: rec.Seq, PrevHash: RecordHash(&rec)})
	return stub.PutState(prefixHead+rec.Principal, newHead)
}

// recordsByPrincipal walks the principal~seq composite index in seq order.
func (c *AuditContract) recordsByPrincipal(ctx contractapi.TransactionContextInterface, principal string) ([]*AuditRecord, error) {
	stub := ctx.GetStub()
	it, err := stub.GetStateByPartialCompositeKey(indexPrinSeq, []string{principal})
	if err != nil {
		return nil, err
	}
	defer it.Close()
	var out []*AuditRecord
	for it.HasNext() {
		kv, err := it.Next()
		if err != nil {
			return nil, err
		}
		raw, err := stub.GetState(prefixRecord + string(kv.Value))
		if err != nil {
			return nil, err
		}
		if raw == nil {
			continue
		}
		var rec AuditRecord
		if err := json.Unmarshal(raw, &rec); err != nil {
			return nil, err
		}
		out = append(out, &rec)
	}
	return out, nil
}

// QueryByPrincipal returns a principal's records in seq order.
func (c *AuditContract) QueryByPrincipal(ctx contractapi.TransactionContextInterface, principal string) ([]*AuditRecord, error) {
	recs, err := c.recordsByPrincipal(ctx, principal)
	if err != nil {
		return nil, err
	}
	if recs == nil {
		recs = []*AuditRecord{}
	}
	return recs, nil
}

// QueryByResource is a CouchDB rich query (needs the index in META-INF).
func (c *AuditContract) QueryByResource(ctx contractapi.TransactionContextInterface, resource string) ([]*AuditRecord, error) {
	q := fmt.Sprintf(`{"selector":{"resource":%s},"use_index":["_design/indexResourceDoc","indexResource"]}`, strconv.Quote(resource))
	it, err := ctx.GetStub().GetQueryResult(q)
	if err != nil {
		return nil, err
	}
	defer it.Close()
	out := []*AuditRecord{}
	for it.HasNext() {
		kv, err := it.Next()
		if err != nil {
			return nil, err
		}
		var rec AuditRecord
		if err := json.Unmarshal(kv.Value, &rec); err != nil {
			continue // composite-index entries and heads are not records
		}
		if rec.RecID != "" {
			out = append(out, &rec)
		}
	}
	return out, nil
}

// VerifyChainRecords is the pure check shared by the transaction and the tests:
// seq must run 1..n, prevHash must equal the hash of the previous record, and
// every signature must verify. Returns the index of the first discontinuity, -1 if intact.
func VerifyChainRecords(pub *ecdsa.PublicKey, recs []*AuditRecord) (int, string) {
	prev := Genesis
	for i, rec := range recs {
		if rec.Seq != int64(i+1) {
			return i, fmt.Sprintf("seq %d at position %d", rec.Seq, i)
		}
		if rec.PrevHash != prev {
			return i, fmt.Sprintf("prevHash mismatch at seq %d", rec.Seq)
		}
		if pub != nil && !VerifySignature(pub, rec) {
			return i, fmt.Sprintf("bad signature at seq %d", rec.Seq)
		}
		prev = RecordHash(rec)
	}
	return -1, ""
}

// VerifyChain recomputes a principal's hash chain from the ledger.
func (c *AuditContract) VerifyChain(ctx contractapi.TransactionContextInterface, principal string) (*VerifyResult, error) {
	recs, err := c.recordsByPrincipal(ctx, principal)
	if err != nil {
		return nil, err
	}
	pub, err := c.pdpKey(ctx)
	if err != nil {
		return nil, err
	}
	idx, reason := VerifyChainRecords(pub, recs)
	if idx == -1 {
		// A truncated tail is internally consistent; the head pointer written at the
		// last accepted LogAccess says how long the chain must be.
		if raw, err := ctx.GetStub().GetState(prefixHead + principal); err == nil && raw != nil {
			var head chainHead
			if json.Unmarshal(raw, &head) == nil && int64(len(recs)) != head.Seq {
				idx = len(recs)
				reason = fmt.Sprintf("head says seq %d but %d records present", head.Seq, len(recs))
			}
		}
	}
	return &VerifyResult{Principal: principal, Records: len(recs), Intact: idx == -1,
		FirstDiscontinuity: idx, Reason: reason}, nil
}

func main() {
	cc, err := contractapi.NewChaincode(&AuditContract{})
	if err != nil {
		panic(err)
	}
	if err := cc.Start(); err != nil {
		panic(err)
	}
}
