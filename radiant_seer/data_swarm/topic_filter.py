"""Topic filter: Iran & Oil focus with geopolitical context.

Headlines: strict Iran/oil filter — only keep directly relevant news.
Contracts: broader filter — keep Iran, oil, AND related geopolitics
(sanctions, Middle East, energy, war) since these markets move together.
"""

from __future__ import annotations

import re

# ── Strict Iran & Oil keywords (for headlines) ──────────────────────
_IRAN_OIL_PATTERNS = [
    # Iran core
    r"\bIran\b", r"\bIranian\b", r"\bTehran\b", r"\bKhamenei\b",
    r"\bIRGC\b", r"\bPasdaran\b", r"\bPersian Gulf\b",
    r"\bStrait of Hormuz\b", r"\bHormuz\b", r"\bKharg\b",
    r"\bBushehr\b", r"\bNatanz\b", r"\bFordow\b",
    r"\bPezeshkian\b", r"\bZarif\b",
    # Iran nuclear
    r"\bJCPOA\b", r"\bnuclear deal\b", r"\buranium\b",
    r"\benrich(?:ment|ing|ed)\b", r"\bcentrifuge\b",
    r"\bIAEA\b", r"\bnuclear weapon\b", r"\bnuclear program\b",
    # Iran proxies
    r"\bHezbollah\b", r"\bHouthi\b", r"\bHamas\b",
    r"\bYemen\b", r"\bRed Sea\b",
    # Oil / energy
    r"\boil\b", r"\bcrude\b", r"\bpetroleum\b", r"\bpetrol\b",
    r"\bBrent\b", r"\bWTI\b", r"\bbarrel[s]?\b",
    r"\bOPEC\b", r"\bOPEC\+\b",
    r"\brefiner(?:y|ies)\b", r"\bpipeline\b",
    r"\bLNG\b", r"\bnatural gas\b", r"\bfuel\b",
    r"\benergy price\b", r"\benergy market\b", r"\benergy crisis\b",
    r"\bgas price\b", r"\bgasoline\b", r"\bdiesel\b",
    r"\btanker\b", r"\boil terminal\b",
    # Precious metals / commodities (strong P&L)
    r"\bgold\b", r"\bGC\)", r"\bsilver\b", r"\bSI\)",
    r"\bCL\)",  # Crude Oil futures ticker
    r"\bprecious metal\b", r"\bbullion\b", r"\bgold price\b",
    r"\bsilver price\b", r"\bplatinum\b", r"\bpalladium\b",
    r"\bCOMEX\b", r"\bgold mine\b", r"\bgold miner\b",
    r"\bsafe.haven\b", r"\bgold reserve\b",
    # Oil geopolitics
    r"\bSaudi\b", r"\bAramco\b", r"\bUAE\b",
    r"\bBahrain\b", r"\bKuwait\b", r"\bQatar\b",
    r"\bGulf state\b",
    # Conflict near oil/Iran
    r"\bblockade\b", r"\bembargo\b",
    r"\bwarship\b", r"\bnavy\b",
    r"\bmissile\b", r"\bdrone strike\b", r"\bairstrike\b",
    r"\bsanction\b",
]

# ── Broader geopolitical context (for contracts) ─────────────────────
_GEO_CONTEXT_PATTERNS = _IRAN_OIL_PATTERNS + [
    # War / conflict
    r"\bwar\b", r"\bceasefire\b", r"\binvasion\b",
    r"\bmilitary\b", r"\bnuclear\b",
    r"\bNATO\b", r"\bUN Security Council\b",
    # Key actors
    r"\bIsrael\b", r"\bNetanyahu\b", r"\bIDF\b",
    r"\bGaza\b", r"\bWest Bank\b", r"\bLebanon\b", r"\bSyria\b",
    r"\bIraq\b", r"\bLibya\b",
    r"\bRussia\b", r"\bPutin\b", r"\bUkraine\b",
    r"\bChina\b", r"\bXi Jinping\b", r"\bTaiwan\b",
    r"\bNorth Korea\b", r"\bKim Jong\b",
    # US foreign policy
    r"\bTrump\b", r"\bBiden\b", r"\bPentagon\b",
    r"\bState Department\b", r"\bCIA\b",
    r"\btariff\b", r"\btrade war\b",
    # Economics that move oil (Fed/Nasdaq excluded — no edge)
    r"\binflation\b", r"\brecession\b", r"\bGDP\b",
    r"\bS&P\s*500\b", r"\bDow\b",
    r"\bbitcoin\b", r"\bcrypto\b",
    r"\bcommodit(?:y|ies)\b",
    # Regime change / leaders
    r"\bregime\b", r"\bcoup\b", r"\bpresident\b.*\bout\b",
    r"\b(?:out|resign|removed)\b.*\bpresident\b",
    r"\belection\b",
]

# ── Noise patterns to always exclude ─────────────────────────────────
_NOISE_PATTERNS = [
    # Sports
    r"\bNHL\b", r"\bNBA\b", r"\bNFL\b", r"\bMLB\b", r"\bMLS\b",
    r"\bStanley Cup\b", r"\bSuper Bowl\b", r"\bWorld Series\b",
    r"\bFIFA\b", r"\bPremier League\b", r"\bChampions League\b",
    r"\bLigue 1\b", r"\bSerie A\b", r"\bBundesliga\b", r"\bLa Liga\b",
    r"\bGrand Slam\b", r"\bUFC\b", r"\bfight.*next\b",
    r"\bO/U\s+\d", r"\bRebounds\b", r"\bAssists\b",  # player prop bets
    # Entertainment
    r"\bGTA\s*VI\b", r"\bGTA\s*6\b",
    r"\bNetflix\b", r"\bTaylor Swift\b", r"\bRihanna\b",
    r"\bPlayboi Carti\b", r"\bBitBoy\b", r"\bThe Weeknd\b",
    r"\bBillboard\b", r"\bHot 100\b",
    r"\bOscar[s]?\b", r"\bGrammy\b", r"\bEmmy\b",
    r"\bvideo\s*game\b", r"\btoken\b", r"\blaunch a token\b",
    # Obscure politics with no signal
    r"\bGuinea.Bissau\b", r"\bMorocco Prime Minister\b",
    r"\bSerbian Parliamentary\b", r"\bThai Prime Minister\b",
    r"\bCyprus House\b", r"\bgubernatorial\b",
    r"\bDemocratic Nominee\b", r"\bRepublican [Nn]ominee\b",
    r"\bDemocratic Party win the \w{2}-\d{2}\b",  # US House seats (NC-05 etc)
    r"\bRepublican Party win the \w{2}-\d{2}\b",
    r"\bNobel Peace Prize\b",
    # Fed/monetary — no edge
    r"\bFOMC\b", r"\bPowell\b", r"\binterest rate\b",
    # Nasdaq — no edge
    r"\bNasdaq\b", r"\bQQQ\b",
    # Gas price — consistent loser
    r"\bgas hit\b", r"\bgasoline\b",
]

_IRAN_OIL_RE = re.compile("|".join(_IRAN_OIL_PATTERNS), re.IGNORECASE)
_GEO_CONTEXT_RE = re.compile("|".join(_GEO_CONTEXT_PATTERNS), re.IGNORECASE)
_NOISE_RE = re.compile("|".join(_NOISE_PATTERNS), re.IGNORECASE)


def is_iran_oil(text: str) -> bool:
    """Return True if text is directly about Iran or oil."""
    return bool(_IRAN_OIL_RE.search(text))


def is_geo_context(text: str) -> bool:
    """Return True if text is geopolitically relevant (broader)."""
    return bool(_GEO_CONTEXT_RE.search(text))


def is_noise(text: str) -> bool:
    """Return True if text is sports/entertainment noise."""
    return bool(_NOISE_RE.search(text))


def filter_headline(text: str) -> bool:
    """Return True if headline should be KEPT.

    Strict: must match Iran/oil keywords.
    """
    if not text.strip():
        return False
    if is_noise(text):
        return False
    return is_iran_oil(text)


def filter_contract(question: str) -> bool:
    """Return True if a Polymarket contract should be KEPT.

    Broader than headlines: keeps Iran, oil, AND surrounding geopolitics
    (wars, sanctions, regime changes, macro events that move oil).
    """
    if not question.strip():
        return False
    if is_noise(question):
        return False
    return is_geo_context(question)
