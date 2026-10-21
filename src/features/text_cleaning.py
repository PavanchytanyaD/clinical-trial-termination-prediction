"""Clean AACT free-text fields and split eligibility criteria into inclusion and exclusion.

AACT flat-file exports encode line breaks as "~" and keep the registry's markdown
escapes (for example "\\>=" for ">="). Text is normalized with NFKC so ligatures and
full-width characters ("ﬁ", "：", "Ⅲ") become plain equivalents.
"""

import re
import unicodedata
from dataclasses import dataclass, field

_MD_ESCAPE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!<>=|~^%&$])")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_SPACES = re.compile(r"[ \t  -​]+")
_BULLET = re.compile(r"^(?:[*\-•·▪◦>]+|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)](?=\s)|\(?[ivx]{1,4}[.)](?=\s))\s*")

# Lines that carry no design information.
_BOILERPLATE = re.compile(
    r"^(?:other\s+)?(?:protocol[- ]defined\s+)?(?:other\s+)?(?:inclusion|exclusion|inclusion/exclusion|eligibility)"
    r"\s+criteri(?:a|on)\s+(?:may\s+)?apply\.?$"
    r"|^(?:further|additional|other)\s+(?:inclusion|exclusion|inclusion/exclusion)\s+criteria\s+(?:may\s+)?apply\.?$",
    re.IGNORECASE,
)
_EMPTY_ITEM = re.compile(r"^(?:none|n/?a|not applicable|nil|-)\.?$", re.IGNORECASE)
_NON_INCLUSION = re.compile(r"non[- ]?inclusion", re.IGNORECASE)


def clean_text(text):
    """Return cleaned text with one line per original line, or "" for missing input."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("~", "\n")
    text = _MD_ESCAPE.sub(r"\1", text)
    text = _CONTROL.sub(" ", text)
    lines = []
    for line in text.split("\n"):
        line = _SPACES.sub(" ", line).strip()
        if line and not _BOILERPLATE.match(_BULLET.sub("", line)):
            lines.append(line)
    return "\n".join(lines)


@dataclass
class Eligibility:
    inclusion: list = field(default_factory=list)
    exclusion: list = field(default_factory=list)
    split_found: bool = False

    @property
    def inclusion_text(self):
        return "\n".join(self.inclusion)

    @property
    def exclusion_text(self):
        return "\n".join(self.exclusion)


def _header_kind(line):
    """Return ("inclusion" | "exclusion", tail) if the line is a section header, else (None, "")."""
    s = _BULLET.sub("", line).strip()
    head, colon, tail = s.partition(":")
    head = head.strip().lower()
    words = head.split()
    if colon:
        if len(words) > 8:
            return None, ""
    elif len(words) > 4 or ("criteri" not in head and head not in ("inclusion", "exclusion", "non-inclusion")):
        return None, ""

    has_exc = "exclusion" in head or bool(_NON_INCLUSION.search(head))
    has_inc = "inclusion" in _NON_INCLUSION.sub("", head)
    if has_inc == has_exc:  # neither, or a combined "inclusion/exclusion" line
        return None, ""
    return ("exclusion" if has_exc else "inclusion"), tail.strip()


def split_eligibility(criteria):
    """Split AACT eligibilities.criteria into inclusion and exclusion items.

    Text before the first header, and all text when no header is found, is treated
    as inclusion; split_found tells the two cases apart.
    """
    result = Eligibility()
    section = "inclusion"
    for line in clean_text(criteria).split("\n"):
        kind, tail = _header_kind(line)
        if kind:
            section = kind
            result.split_found = True
            line = tail
        item = _BULLET.sub("", line).strip()
        if item and not _EMPTY_ITEM.match(item):
            getattr(result, section).append(item)
    return result
