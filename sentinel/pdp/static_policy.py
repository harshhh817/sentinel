"""Static RBAC/ABAC entitlement -- line 1 of Algorithm 1.

Stands in for IAM in local mode. A JSON policy maps principals to roles, roles to
allow statements (glob-matched actions and resources, optional ABAC conditions), and
resource patterns to a sensitivity label. Deny-by-default: anything not explicitly
allowed is forbidden, and a static denial is final regardless of R, so Sentinel can
only ever be more restrictive than the underlying entitlement, never less.
"""

from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrincipalInfo:
    role: str
    unit: str


class StaticPolicy:
    def __init__(self, doc: dict):
        self.doc = doc
        self.default_role = doc.get("default_role", "user")
        self.principals = {k: v for k, v in doc.get("principals", {}).items()
                           if not k.startswith("_")}
        self.roles = doc.get("roles", {})
        sens = doc.get("sensitivity", {})
        self.sens_rules = [(r["pattern"], float(r["value"])) for r in sens.get("rules", [])]
        self.sens_default = float(sens.get("default", 0.5))

    @classmethod
    def load(cls, path: Path) -> StaticPolicy:
        return cls(json.loads(Path(path).read_text()))

    def principal(self, principal: str) -> PrincipalInfo:
        p = self.principals.get(principal, {})
        return PrincipalInfo(role=p.get("role", self.default_role), unit=p.get("unit", ""))

    def org(self, principal: str) -> dict[str, str]:
        """The org dict the feature builder expects (subset of the LDAP profile)."""
        info = self.principal(principal)
        return {"role": info.role, "functional_unit": info.unit, "business_unit": info.unit}

    def own_bucket(self, principal: str) -> str:
        unit = self.principal(principal).unit
        return f"sentinel-{unit}" if unit else ""

    # --- entitlement -------------------------------------------------------

    def permitted(self, principal: str, action: str, resource: str,
                  context: dict | None = None) -> tuple[bool, str]:
        """(allowed, reason). Deny-by-default."""
        info = self.principal(principal)
        role = self.roles.get(info.role)
        if role is None:
            return False, f"unknown role {info.role!r}"
        for stmt in role.get("allow", []):
            if not any(fnmatch.fnmatchcase(action, a) for a in stmt.get("actions", [])):
                continue
            if not any(fnmatch.fnmatchcase(resource, r) for r in stmt.get("resources", [])):
                continue
            ok, why = self._conditions(stmt.get("condition", {}), principal, resource,
                                       context or {})
            if ok:
                return True, f"role {info.role}"
            return False, why
        return False, f"no statement in role {info.role!r} allows {action} on {resource}"

    def _conditions(self, cond: dict, principal: str, resource: str,
                    context: dict) -> tuple[bool, str]:
        if cond.get("own_unit_bucket"):
            own = self.own_bucket(principal)
            bucket = resource.split(":::", 1)[1].split("/", 1)[0] if ":::" in resource else ""
            if not own or bucket != own:
                return False, f"bucket {bucket!r} is not the principal's unit bucket {own!r}"
        if cond.get("require_managed_device") and not context.get("device_managed"):
            return False, "managed device required"
        return True, ""

    # --- sensitivity -------------------------------------------------------

    def sensitivity(self, resource: str) -> float:
        for pattern, value in self.sens_rules:
            if fnmatch.fnmatchcase(resource, pattern):
                return value
        return self.sens_default
