"""Tests for the Iran & Oil topic filter."""

from radiant_seer.data_swarm.topic_filter import (
    filter_contract,
    filter_headline,
    is_iran_oil,
)


class TestIsIranOil:
    def test_iran_keywords(self):
        assert is_iran_oil("Iran threatens retaliation after airstrike")
        assert is_iran_oil("Tehran warns of Strait of Hormuz closure")
        assert is_iran_oil("IRGC launches missile test")
        assert is_iran_oil("JCPOA nuclear deal negotiations resume")
        assert is_iran_oil("Uranium enrichment reaches 60%")
        assert is_iran_oil("Khamenei issues statement on sanctions")

    def test_oil_keywords(self):
        assert is_iran_oil("Oil prices surge past $100 per barrel")
        assert is_iran_oil("OPEC+ agrees to cut crude production")
        assert is_iran_oil("Brent crude hits $95")
        assert is_iran_oil("WTI drops on demand fears")
        assert is_iran_oil("Natural gas prices spike in Europe")
        assert is_iran_oil("LNG exports reach record high")
        assert is_iran_oil("Refinery explosion in Texas")

    def test_iran_oil_intersection(self):
        assert is_iran_oil("Iran oil sanctions tightened by US")
        assert is_iran_oil("Tanker seized in Strait of Hormuz")
        assert is_iran_oil("Saudi Aramco increases production amid Iran tensions")
        assert is_iran_oil("Houthi drone strikes oil terminal")

    def test_proxy_groups(self):
        assert is_iran_oil("Hezbollah rocket barrage intensifies")
        assert is_iran_oil("Houthi attacks in Red Sea disrupt shipping")
        assert is_iran_oil("Hamas ceasefire talks collapse")

    def test_not_iran_oil(self):
        assert not is_iran_oil("Lakers vs Celtics game tonight")
        assert not is_iran_oil("New Taylor Swift album drops Friday")
        assert not is_iran_oil("GTA VI release date confirmed")
        assert not is_iran_oil("Will the Hurricanes win the Stanley Cup?")
        assert not is_iran_oil("2028 Democratic presidential nomination")


class TestFilterHeadline:
    def test_keeps_iran(self):
        assert filter_headline("Iran threatens to close Strait of Hormuz")
        assert filter_headline("IAEA inspectors denied access to Natanz")

    def test_keeps_oil(self):
        assert filter_headline("Oil prices spike after OPEC cuts")
        assert filter_headline("Crude oil inventory report shows drawdown")

    def test_drops_unrelated(self):
        assert not filter_headline("NFL playoff schedule announced")
        assert not filter_headline("New iPhone features leaked")
        assert not filter_headline("")

    def test_keeps_gulf_conflict(self):
        assert filter_headline("US warships deployed to Persian Gulf")
        assert filter_headline("Drone strike hits UAE oil terminal")


class TestFilterContract:
    def test_keeps_iran_contracts(self):
        assert filter_contract("Will the US strike Iran by June 2026?")
        assert filter_contract("Will Iran enrich uranium to 90%?")
        assert filter_contract("Strait of Hormuz blocked before 2027?")

    def test_keeps_oil_contracts(self):
        assert filter_contract("Will oil hit $150 per barrel in 2026?")
        assert filter_contract("OPEC production cut before July?")
        assert filter_contract("Will Brent crude exceed $120?")

    def test_drops_unrelated_contracts(self):
        assert not filter_contract("Will the Lakers win the NBA Finals?")
        assert not filter_contract("GTA VI released before June 2026?")
        assert not filter_contract("Will Gavin Newsom win the 2028 Dem nomination?")
