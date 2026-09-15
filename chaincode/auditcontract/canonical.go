// Package main implements the Sentinel audit contract (Section IV-E of the paper).
//
// canonical.go is a byte-exact port of sentinel/pdp/signer.py: the ECDSA signature made by
// the PDP in Python must verify here, so the canonical encoding of the record tuple
// has to produce identical bytes. Rules: the eleven signed fields in sorted key order,
// no whitespace, floats rounded to 8 decimals and printed the way Python's repr()
// prints them, null for missing r/R, strings escaped as json.dumps(ensure_ascii=True).
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// AuditRecord is the tuple of eq. (5) plus the PDP signature.
type AuditRecord struct {
	RecID     string   `json:"recId"`
	PrevHash  string   `json:"prevHash"`
	Seq       int64    `json:"seq"`
	TS        string   `json:"ts"`
	Principal string   `json:"principal"`
	Action    string   `json:"action"`
	Resource  string   `json:"resource"`
	R         any      `json:"r"` // float64 or nil (static deny); contractapi forbids *float64
	RR        any      `json:"R"`
	Verdict   string   `json:"verdict"`
	HFeat     string   `json:"h_feat"`
	Sig       string   `json:"sig"`
}

// pyRepr formats a float the way Python's json.dumps does after round(v, 8).
func pyRepr(v float64) string {
	v = math.Round(v*1e8) / 1e8
	if v == 0 {
		return "0.0"
	}
	if v == math.Trunc(v) && math.Abs(v) < 1e16 {
		return strconv.FormatFloat(v, 'f', 1, 64)
	}
	exp := math.Floor(math.Log10(math.Abs(v)))
	if exp < -4 || exp >= 16 {
		// Python: 1e-05, 1.5e-05, 1e+16 -- two-digit exponent minimum.
		s := strconv.FormatFloat(v, 'e', -1, 64) // e.g. 1.5e-05 or 1e-05
		return s
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// pyString escapes like json.dumps(ensure_ascii=True): ", \, control chars, non-ASCII as \uXXXX.
func pyString(s string) string {
	var b strings.Builder
	b.WriteByte('"')
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		switch {
		case r == '"':
			b.WriteString(`\"`)
		case r == '\\':
			b.WriteString(`\\`)
		case r == '\n':
			b.WriteString(`\n`)
		case r == '\r':
			b.WriteString(`\r`)
		case r == '\t':
			b.WriteString(`\t`)
		case r == '\b':
			b.WriteString(`\b`)
		case r == '\f':
			b.WriteString(`\f`)
		case r < 0x20 || r > 0x7e:
			if r > 0xffff {
				r -= 0x10000
				b.WriteString(fmt.Sprintf(`\u%04x\u%04x`, 0xd800+(r>>10), 0xdc00+(r&0x3ff)))
			} else {
				b.WriteString(fmt.Sprintf(`\u%04x`, r))
			}
		default:
			b.WriteRune(r)
		}
		i += size
	}
	b.WriteByte('"')
	return b.String()
}

func pyFloatOrNull(v any) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case float64:
		return pyRepr(x)
	case float32:
		return pyRepr(float64(x))
	case int:
		return pyRepr(float64(x))
	case int64:
		return pyRepr(float64(x))
	case json.Number:
		f, err := x.Float64()
		if err != nil {
			return "null"
		}
		return pyRepr(f)
	default:
		return "null"
	}
}

// Canonical returns the signed bytes: sorted keys, no whitespace.
// Key order (Python sort_keys, code-point order): R, action, h_feat, prevHash,
// principal, r, recId, resource, seq, ts, verdict.
func Canonical(rec *AuditRecord) []byte {
	var b strings.Builder
	b.WriteString(`{"R":` + pyFloatOrNull(rec.RR))
	b.WriteString(`,"action":` + pyString(rec.Action))
	b.WriteString(`,"h_feat":` + pyString(rec.HFeat))
	b.WriteString(`,"prevHash":` + pyString(rec.PrevHash))
	b.WriteString(`,"principal":` + pyString(rec.Principal))
	b.WriteString(`,"r":` + pyFloatOrNull(rec.R))
	b.WriteString(`,"recId":` + pyString(rec.RecID))
	b.WriteString(`,"resource":` + pyString(rec.Resource))
	b.WriteString(`,"seq":` + strconv.FormatInt(rec.Seq, 10))
	b.WriteString(`,"ts":` + pyString(rec.TS))
	b.WriteString(`,"verdict":` + pyString(rec.Verdict))
	b.WriteByte('}')
	return []byte(b.String())
}

// RecordHash mirrors sentinel.pdp.chain.record_hash: SHA-256(canonical || sig).
func RecordHash(rec *AuditRecord) string {
	h := sha256.New()
	h.Write(Canonical(rec))
	h.Write([]byte(rec.Sig))
	return hex.EncodeToString(h.Sum(nil))
}

// Genesis is the prevHash of a principal's first record.
const Genesis = "0000000000000000000000000000000000000000000000000000000000000000"
