"""Sanctions dataset constants for OpenSanctions FtM data.

Data license: OpenSanctions data is CC BY-NC 4.0.
  https://www.opensanctions.org/licensing/
Commercial use of the data (or outputs derived from it) requires a
separate license from OpenSanctions. This code (MIT) is independent
of the data license.
"""

OPENSANCTIONS_URL = "https://data.opensanctions.org/datasets/latest/default/targets.nested.json"

# 55 confirmed sanctions datasets from profiling OpenSanctions default collection.
# Identified via keyword matching (sanctions, sdn, ofac, terror, freeze, etc.)
# plus explicit includes for known sources. Excludes PEP lists, Wikidata,
# wanted lists, and exclusion/debarment lists that aren't sanctions.
DEFAULT_SANCTIONS_DATASETS: frozenset[str] = frozenset({
    "adb_sanctions",
    "ae_local_terrorists",
    "afdb_sanctions",
    "at_nbter_sanctions",
    "au_dfat_sanctions",
    "az_fiu_sanctions",
    "be_fod_sanctions",
    "ca_dfatd_sema_sanctions",
    "ch_seco_sanctions",
    "cn_sanctions",
    "cz_national_sanctions",
    "cz_terrorists",
    "ee_international_sanctions",
    "eg_terrorists",
    "eu_cor_members",
    "eu_fsf",
    "eu_journal_sanctions",
    "eu_sanctions_map",
    "ext_us_ofac_press_releases",
    "gb_fcdo_sanctions",
    "gb_hmt_sanctions",
    "iadb_sanctions",
    "il_mod_terrorists",
    "il_wmd_sanctions",
    "ir_sanctions",
    "jo_sanctions",
    "jp_mof_sanctions",
    "kg_fiu_national",
    "kz_afmrk_sanctions",
    "lt_fiu_freezes",
    "lv_fiu_sanctions",
    "mc_fund_freezes",
    "md_terror_sanctions",
    "my_moha_sanctions",
    "ng_nigsac_sanctions",
    "nl_terrorism_list",
    "np_mha_sanctions",
    "nz_russia_sanctions",
    "ph_amlc_sanctions",
    "pl_finanse_sanctions",
    "pl_mswia_sanctions",
    "qa_nctc_sanctions",
    "ro_onpcsb_sanctions",
    "ru_mfa_sanctions",
    "sg_terrorists",
    "th_designated_person",
    "ua_nsdc_sanctions",
    "ua_war_sanctions",
    "un_sc_sanctions",
    "us_bis_denied",
    "us_ofac_cons",
    "us_ofac_enforcement_actions",
    "us_ofac_sdn",
    "us_trade_csl",
    "za_fic_sanctions",
})
