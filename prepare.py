"""
Option Pricing Tuning – Data & Evaluation Pipeline (Real Data Edition)
======================================================================
Uses REAL stock prices from Yahoo Finance with synthetic options generated
on those real prices. Vol surface parameters are derived from each stock's
actual realized volatility.

This tests the model against real market dynamics (crashes, rallies, regime
changes) while keeping the evaluation framework intact.
"""

import time
from pathlib import Path

import numpy as np
from scipy.stats import norm
from joblib import Parallel, delayed

# ─── Configuration ────────────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".cache" / "option-pricing-tuning"
DATA_FILE = CACHE_DIR / "market_data_real.npz"
TIME_BUDGET = 120  # seconds per experiment run
SEED = 42

# S&P 500 + Russell 2000 coverage (~2600 US stocks)
TICKERS = [
    "A", "AAL", "AAOI", "AAON", "AAPL", "ABBV", "ABCL", "ABEO",
    "ABNB", "ABOS", "ABT", "ABUS", "ACAD", "ACGL",
    "ACHC", "ACIW", "ACLS", "ACLX", "ACMR", "ACN", "ACNB",
    "ACRS", "ACTG", "ADAG", "ADBE", "ADEA", "ADI", "ADM", "ADMA", "ADP",
    "ADPT", "ADSK", "ADTN", "ADUS", "AEE",
    "AEHR", "AEIS", "AEP", "AES", "AEVA", "AEYE",
    "AFL", "AFRM", "AFYA", "AGEN", "AGIO", "AGNC", "AGYS", "AHCO",
    "AIG", "AIOT", "AIZ", "AJG", "AKAM", "AKBA",
    "ALAB", "ALB", "ALCO", "ALDX", "ALGM", "ALGN", "ALGT", "ALHC", "ALKS", "ALKT", "ALL",
    "ALLE", "ALLO", "ALLT", "ALNY", "ALOT", "ALRM", "ALRS",
    "ALT", "ALTI", "ALTO", "AMAL", "AMAT", "AMBA", "AMCR",
    "AMCX", "AMD", "AME", "AMGN", "AMKR", "AMLX", "AMP", "AMPH", "AMPL", "AMRN", "AMRX",
    "AMSC", "AMSF", "AMT", "AMWD", "AMZN", "ANAB", "ANDE", "ANET", "ANGI", "ANGO", "ANIK",
    "ANIP", "AON", "AOS", "AOSL", "AOUT",
    "APA", "APD", "APEI", "APGE", "APH", "API", "APLD", "APLS", "APO", "APOG", "APP",
    "APPF", "APPN", "APPS", "APTV", "AQST", "ARBE", "ARCB", "ARCC", "ARCT", "ARDX", "ARE",
    "ARES", "ARGX", "ARHS", "ARKO", "ARLP", "ARM", "AROW", "ARQT", "ARRY",
    "ARVN", "ARWR", "ASMB", "ASML", "ASND", "ASO",
    "ASRT", "ASTE", "ASTL", "ASTS", "ASUR",
    "ATEC", "ATEX", "ATLC", "ATLO", "ATNI", "ATO", "ATRC", "ATRO",
    "AUPH", "AUR", "AURE", "AVAV", "AVB",
    "AVGO", "AVIR", "AVNW", "AVO", "AVPT", "AVT", "AVXL", "AVY",
    "AWK", "AXGN", "AXON", "AXP", "AXSM", "AXTI", "AZO", "AZTA", "BA", "BAC",
    "BALL", "BAND", "BANF", "BANR", "BAX", "BBIO",
    "BBSI", "BBY", "BCBP", "BCPC", "BCRX", "BCYC",
    "BDX", "BEAM", "BEAT", "BEN", "BF-B",
    "BFST", "BG", "BGC", "BHF", "BHRB",
    "BIIB", "BILI", "BJRI", "BK", "BKNG", "BKR", "BL", "BLBD", "BLDP", "BLDR", "BLFS",
    "BLK", "BLKB", "BLMN", "BLNK", "BMBL", "BMEA",
    "BMRC", "BMRN", "BMY", "BNC", "BNTX",
    "BOKF", "BOOM", "BPOP", "BR",
    "BRZE", "BSBK", "BSRR", "BSX", "BSY", "BTDR",
    "BUSE", "BX", "BXP", "BYND", "BYRN", "BZ", "BZUN", "C",
    "CACC", "CAG", "CAH", "CAKE", "CALM", "CAMP", "CAMT",
    "CAR", "CARE", "CARG", "CARR", "CART", "CASH", "CASS", "CASY", "CAT",
    "CATY", "CB", "CBNK", "CBOE", "CBRE", "CBRL",
    "CBSH", "CCBG", "CCC", "CCEP", "CCI", "CCL",
    "CCNE", "CCOI", "CCRN", "CCSI", "CDNA", "CDNS", "CDW",
    "CECO", "CEG", "CELH", "CENT", "CENX", "CERS", "CEVA", "CF",
    "CFFI", "CFG", "CFLT", "CG", "CGBD", "CGEN", "CGNX",
    "CHCO", "CHD", "CHDN", "CHEF", "CHKP", "CHMG", "CHRD", "CHRS",
    "CHRW", "CHTR", "CI", "CIEN", "CIFR", "CIGI", "CINF", "CIVB", "CL",
    "CLAR", "CLBT", "CLDX", "CLFD", "CLMB", "CLMT", "CLNE",
    "CLSK", "CLX", "CMCO", "CMCSA", "CME", "CMG", "CMI", "CMPR",
    "CMS", "CMTL", "CNC", "CNDT", "CNOB", "CNP",
    "COCO", "COF", "COFS", "COHU", "COIN", "COKE",
    "COLB", "COLL", "COLM", "COO", "COP", "COR", "CORT", "CORZ", "COST", "CPAY", "CPB",
    "CPRT", "CPRX", "CPSS", "CPT", "CRAI",
    "CRDO", "CRH", "CRL", "CRM", "CRMD", "CRMT", "CRNC",
    "CRNT", "CRNX", "CRON", "CROX", "CRSP", "CRSR", "CRTO", "CRUS", "CRVL", "CRWD",
    "CSBR", "CSCO", "CSGP", "CSGS", "CSIQ", "CSPI",
    "CSWC", "CSX", "CTAS", "CTBI",
    "CTLP", "CTMX", "CTRA", "CTRN", "CTSH", "CTVA",
    "CVBF", "CVCO", "CVGW", "CVLT", "CVNA", "CVRX", "CVS",
    "CVX", "CWBC", "CWST", "CYRX", "CYTK", "CZR", "D",
    "DAKT", "DAL", "DASH", "DAVE", "DDOG", "DE", "DECK", "DELL", "DFS",
    "DG", "DGII", "DGX", "DHI", "DHR", "DIBS", "DIOD", "DIS",
    "DKNG", "DLO", "DLR", "DLTR",
    "DNUT", "DOC", "DOCU", "DOMO", "DORM", "DOV", "DOW", "DOX",
    "DPZ", "DRH", "DRI", "DRIO", "DRS", "DRVN", "DSGR",
    "DSP", "DTE", "DTIL", "DUK", "DUOL",
    "DVA", "DVN", "DXCM", "DXPE", "DYN", "EA", "EBAY", "EBC",
    "ECL", "ECPG", "ED", "EEFT", "EFSC", "EFX", "EG",
    "EGBN", "EIX", "EL", "ELTK", "ELV",
    "EME", "EMR", "ENLT", "ENPH", "ENSG",
    "ENTA", "ENTG", "ENVX", "EOG", "EOSE", "EPAM", "EQIX",
    "EQR", "EQT", "ERIC", "ERIE", "ERII", "ES", "ESCA",
    "ESLT", "ESPR", "ESS", "ETN", "ETON", "ETR",
    "EVCM", "EVER", "EVGO", "EVRG", "EW", "EWBC", "EWTX", "EXAS", "EXC", "EXE", "EXEL",
    "EXLS", "EXPD", "EXPE", "EXPI", "EXPO", "EXR", "EXTR", "EYE", "EYPT", "F",
    "FANG", "FAST", "FATE", "FBIZ", "FBNC", "FCFS", "FCNCA", "FCX",
    "FDS", "FDX", "FE", "FELE", "FFBC", "FFIC", "FFIN", "FFIV",
    "FHB", "FIBK", "FICO", "FIS", "FISI", "FISV", "FITB", "FIVE",
    "FIVN", "FIX", "FIZZ", "FLEX", "FLGT", "FLNC", "FLNT", "FLWS",
    "FLXS", "FLYW", "FMBH", "FMNB", "FNKO", "FNLC",
    "FOLD", "FONR", "FORM", "FORR", "FORTY", "FOX", "FOXA", "FOXF",
    "FRME", "FROG", "FRPH", "FRPT", "FRSH", "FRST", "FRT", "FSBW",
    "FSLR", "FSLY", "FSTR", "FSV", "FTAI", "FTDR", "FTNT",
    "FTV", "FULC", "FULT", "FWRD",
    "FWRG", "GABC", "GAIA", "GAIN", "GANX",
    "GBDC", "GD", "GDDY", "GDEN", "GDRX", "GDS", "GDYN",
    "GE", "GEHC", "GEN", "GEOS", "GERN", "GEV", "GEVO",
    "GH", "GIII", "GILD", "GILT", "GIS", "GL", "GLAD", "GLBE", "GLDD",
    "GLNG", "GLPI", "GLRE", "GLW", "GM", "GMAB",
    "GNRC", "GNTX", "GO", "GOGO", "GOOD", "GOOG", "GOOGL",
    "GPC", "GPCR", "GPN", "GPRE", "GPRO", "GRAB",
    "GRFS", "GRMN", "GRPN", "GRVY", "GS", "GSAT", "GSBC", "GSHD",
    "GT", "GTLB", "GTX", "GWRS", "GWW",
    "HAFC", "HAIN", "HAL", "HALO", "HAS", "HBAN",
    "HBCP", "HBNC", "HBT", "HCA", "HCAT", "HCKT",
    "HCSG", "HD", "HDSN", "HELE", "HELP",
    "HFWA", "HIFS", "HIG", "HII", "HIMX",
    "HLIT", "HLMN", "HLNE", "HLT", "HNNA", "HNST",
    "HOLX", "HON", "HOOD", "HOPE", "HPE", "HPK", "HPQ",
    "HQY", "HRL", "HRMY", "HROW", "HRTX", "HRZN", "HSIC", "HST", "HSTM", "HSY",
    "HTBK", "HTLD", "HTZ", "HUBB", "HUBG", "HUM",
    "HURN", "HUT", "HWBK", "HWC", "HWKN", "HWM",
    "IAC", "IART", "IBCP", "IBEX", "IBKR", "IBM", "IBOC", "ICE",
    "ICFI", "ICHR", "ICLR", "ICUI", "IDCC", "IDXX", "IDYA",
    "IESC", "IEX", "IFF", "IHRT", "IIIV",
    "ILMN", "IMCR", "IMDX", "IMMR", "IMNM", "IMRX", "IMVT",
    "IMXI", "INBK", "INCY", "INDB", "INDI", "INGN",
    "INMD", "INO", "INOD", "INSM", "INTC", "INTR", "INTU",
    "INVA", "INVH", "IONS", "IOSP", "IOVA", "IP", "IPAR", "IPGP",
    "IQV", "IR", "IRDM", "IREN", "IRM", "IRMD",
    "IRTC", "IRWD", "ISRG", "IT", "ITIC", "ITRI", "ITRN", "ITW",
    "IVZ", "J", "JACK", "JAKK", "JANX", "JAZZ", "JBHT", "JBL",
    "JBLU", "JBSS", "JCI", "JD", "JJSF", "JKHY",
    "JNJ", "JOUT", "JPM", "JRVR", "JYNT",
    "KALU", "KALV", "KC", "KDP", "KE", "KEY", "KEYS", "KHC",
    "KIM", "KINS", "KKR", "KLAC", "KLIC", "KLTR", "KMB", "KMI",
    "KNSA", "KO", "KOPN", "KPTI", "KR", "KRMD", "KRNT", "KRNY", "KROS",
    "KRT", "KRUS", "KRYS", "KSPI", "KTOS", "KURA", "KVHI", "KVUE", "KYMR",
    "L", "LAKE", "LAMR", "LAND", "LARK", "LASR", "LAUR",
    "LCID", "LCNB", "LCUT", "LDOS", "LECO", "LEE", "LEGH", "LEGN", "LEN",
    "LENZ", "LFMD", "LFST", "LFUS", "LGIH", "LGND", "LH",
    "LHX", "LI", "LII", "LIN", "LINC", "LINE",
    "LITE", "LIVN", "LKFN", "LKQ", "LLY", "LMAT", "LMB", "LMNR", "LMT",
    "LNT", "LNTH", "LOCO", "LOGI", "LOOP", "LOPE", "LOVE", "LOW",
    "LPLA", "LPRO", "LQDT", "LRCX", "LSAK", "LSBK", "LSCC",
    "LSTR", "LTRX", "LULU", "LUNG", "LUNR", "LUV", "LVS", "LW",
    "LWLG", "LXRX", "LYB", "LYFT", "LYTS", "LYV", "LZ", "MA",
    "MAA", "MANH", "MAR", "MARA", "MAS", "MASI", "MAT",
    "MATW", "MB", "MBIN", "MBLY", "MBUU", "MBWM",
    "MCD", "MCFT", "MCHP", "MCK", "MCO", "MCRB", "MCRI",
    "MDB", "MDGL", "MDLZ", "MDT", "MDXG",
    "MEDP", "MELI", "MEOH", "MERC", "MET", "META", "METC",
    "MFIC", "MFIN", "MGEE", "MGM", "MGNI", "MGNX", "MGPI",
    "MGTX", "MIDD", "MIND", "MIRM", "MITK", "MKC", "MKSI",
    "MKTX", "MLAB", "MLKN", "MLM", "MLTX", "MMM",
    "MMSI", "MNDY", "MNKD", "MNRO", "MNSB", "MNST", "MO",
    "MOH", "MOLN", "MORN", "MOS", "MPAA", "MPC", "MPWR",
    "MQ", "MRBK", "MRCY", "MRK", "MRNA", "MRTN",
    "MRVI", "MRVL", "MS", "MSBI", "MSCI", "MSEX", "MSFT", "MSI", "MSTR", "MTB",
    "MTCH", "MTD", "MTLS", "MTRX", "MTSI", "MU", "MVBF", "MVIS",
    "MXL", "MYFW", "MYGN", "MYRG", "NATH",
    "NATR", "NAVI", "NBBK", "NBIX", "NBTB",
    "NCLH", "NCMI", "NCNO", "NDAQ", "NDSN", "NEE", "NEM", "NEO",
    "NEOG", "NEOV", "NEWT", "NFLX", "NI", "NICE",
    "NIU", "NKE", "NKTR", "NKTX", "NMFC", "NMIH", "NMRA",
    "NMRK", "NNBR", "NNDM", "NNOX", "NOC", "NODK", "NOVT", "NOW",
    "NRG", "NRIM", "NRIX", "NSC", "NSIT", "NSSC",
    "NTAP", "NTCT", "NTES", "NTGR", "NTLA", "NTNX", "NTRA", "NTRS",
    "NUE", "NUTX", "NUVL", "NVAX", "NVCR", "NVDA", "NVEC", "NVMI",
    "NVR", "NVTS", "NWBI", "NWE", "NWFL", "NWL", "NWPX", "NWS", "NWSA", "NXPI", "NXST",
    "NXT", "O", "OCFC", "OCGN",
    "OCSL", "OCUL", "ODFL", "OFIX", "OFLX",
    "OKE", "OKTA", "OLED", "OLLI", "OLPX", "OM", "OMC", "OMCL",
    "OMER", "ON", "ONB", "ONEW",
    "OPBK", "OPCH", "OPEN", "OPK", "OPRT",
    "ORCL", "ORIC", "ORLY", "ORRF", "OSBC", "OSIS",
    "OSPN", "OSS", "OSUR", "OTIS", "OTLY",
    "OTTR", "OUST", "OVID", "OVLY", "OXY", "OZK", "PACB",
    "PAHC", "PAMT", "PANW", "PARK", "PATK", "PAYC", "PAYO",
    "PAYS", "PAYX", "PBHC", "PCAR", "PCG", "PCRX", "PCT", "PCTY",
    "PCVX", "PDD", "PDEX", "PDFS",
    "PEBK", "PEBO", "PECO", "PEG", "PEGA",
    "PENN", "PEP", "PERI", "PFBC", "PFE", "PFG", "PFIS", "PG",
    "PGEN", "PGNY", "PGR", "PH", "PHAT", "PHM",
    "PI", "PKBK", "PKG", "PKOH", "PLAB", "PLAY", "PLCE", "PLD", "PLMR",
    "PLPC", "PLRX", "PLSE", "PLTR", "PLUG", "PLUS", "PLXS", "PM",
    "PMVP", "PNC", "PNR", "PNRG", "PNTG", "PNW", "PODD", "POOL", "POWI",
    "POWL", "POWW", "PPC", "PPG", "PPL", "PRAA", "PRCH",
    "PRDO", "PRGS", "PRLD", "PROF", "PROV", "PRPL",
    "PRTA", "PRTC", "PRTH", "PRTS", "PRU", "PRVA", "PSA", "PSEC",
    "PSMT", "PSX", "PTC", "PTCT", "PTEN", "PTGX", "PTLO",
    "PTON", "PUBM", "PWP", "PWR", "PYPL", "PYXS", "PZZA", "QCOM", "QCRH",
    "QDEL", "QFIN", "QLYS", "QNST", "QRVO", "QS",
    "QUBT", "QURE", "RAIL", "RANI", "RARE",
    "RBB", "RBBN", "RCAT", "RCEL", "RCKT",
    "RCKY", "RCL", "RCMT", "RDNT", "RDVT", "RDWR",
    "REAL", "REG", "REGN", "REKR", "RELL", "RELY", "RENT", "REPL", "REYN", "RF",
    "RGCO", "RGEN", "RGLD", "RGNX", "RGP", "RGTI", "RICK", "RIGL", "RILY", "RIOT",
    "RIVN", "RJF", "RKLB", "RL", "RLAY", "RMBS", "RMD", "RMNI",
    "RMR", "RNA", "RNAC", "ROAD", "ROCK", "ROIV", "ROK", "ROKU", "ROL",
    "ROOT", "ROP", "ROST", "RPAY", "RPD", "RPRX", "RRBI", "RRGB", "RRR", "RSG",
    "RTX", "RUM", "RUN", "RUSHA", "RVMD", "RVSB", "RVTY",
    "RXRX", "RXST", "RYTM", "SABR", "SAFT", "SAIA", "SAIC",
    "SAIL", "SAMG", "SANA", "SANM", "SATS", "SBAC", "SBCF",
    "SBFG", "SBGI", "SBLK", "SBRA", "SBUX", "SCHL", "SCHW",
    "SCSC", "SCVL", "SDGR", "SEDG", "SEER", "SEIC",
    "SENS", "SERA", "SERV", "SEVN", "SEZL", "SFBC",
    "SFM", "SFNC", "SFST", "SGC", "SGHT", "SGML", "SGMO", "SGMT",
    "SGRY", "SHBI", "SHC", "SHEN", "SHLS", "SHOO", "SHOP", "SHW",
    "SIGA", "SIGI", "SILC", "SIMO", "SIRI", "SITM", "SJM",
    "SKIN", "SKWD", "SKYW", "SLAB", "SLB", "SLDP",
    "SLM", "SLNG", "SLNO", "SLP", "SLRC",
    "SMBC", "SMCI", "SMMT", "SMPL", "SMTC", "SMTI",
    "SNA", "SNBR", "SNCY", "SNDL", "SNDX", "SNEX", "SNPS",
    "SO", "SOFI", "SOHU", "SONO", "SOUN",
    "SPFI", "SPG", "SPGI", "SPOK", "SPRO", "SPSC", "SPT",
    "SRAD", "SRCE", "SRE", "SRPT", "SRRK", "SSBI", "SSNC", "SSP",
    "SSRM", "SSSS", "SSYS", "STAA", "STBA", "STE", "STEP",
    "STGW", "STIM", "STKL", "STKS", "STLD", "STNE",
    "STRA", "STRD", "STRL", "STRO", "STRS", "STRT",
    "STT", "STTK", "STX", "STZ", "SUPN",
    "SVC", "SW", "SWBI", "SWK", "SWKH", "SWKS", "SY", "SYBT",
    "SYF", "SYK", "SYM", "SYNA", "SYY", "T", "TACO", "TALK", "TAP",
    "TARS", "TASK", "TBBK", "TBLA", "TBPH",
    "TCBI", "TCBK", "TCMD", "TCOM", "TCPC",
    "TDG", "TDY", "TEAM", "TECH",
    "TEL", "TEM", "TENB", "TER", "TERN", "TFC", "TFSL", "TGT", "TGTX",
    "THFF", "THRM", "THRY", "TIGR", "TILE", "TIPT", "TITN", "TJX",
    "TKO", "TLN", "TLRY", "TLS", "TMDX", "TMO", "TMUS", "TNDM",
    "TNGX", "TNXP", "TOWN", "TPG", "TPL", "TPR",
    "TREE", "TRGP", "TRIN", "TRIP", "TRMB", "TRMD", "TRMK", "TRNS",
    "TROW", "TRS", "TRST", "TRUP", "TRV",
    "TSBK", "TSCO", "TSEM", "TSLA", "TSN", "TT", "TTD", "TTEC", "TTEK",
    "TTMI", "TTWO", "TVTX", "TW", "TWFG", "TWIN", "TWST", "TXG", "TXN", "TXRH", "TXT",
    "TYL", "TYRA", "UAL", "UBER", "UBSI",
    "UCTT", "UDMY", "UDR", "UEIC", "UFCS", "UFPI", "UFPT",
    "UHS", "ULCC", "ULH", "ULTA", "UMBF",
    "UNH", "UNIT", "UNP", "UNTY", "UPBD", "UPS", "UPST", "UPWK",
    "URBN", "URGN", "URI", "USB", "USLM", "UTHR", "UTMD",
    "UVSP", "V", "VALU", "VCEL", "VCTR", "VCYT",
    "VECO", "VERA", "VERI", "VERX",
    "VFC", "VIAV", "VICI", "VICR", "VIR", "VIRC", "VITL", "VKTX",
    "VLO", "VLTO", "VLY", "VMC", "VNDA", "VNET", "VNOM",
    "VRA", "VREX", "VRM", "VRNS", "VRRM", "VRSK", "VRSN", "VRTX", "VSAT", "VSEC", "VST",
    "VSTM", "VTR", "VTRS", "VUZI", "VYGR", "VZ", "WAB", "WABC", "WAFD",
    "WASH", "WAT", "WAY", "WBD", "WDAY", "WDC", "WDFC", "WEC",
    "WELL", "WEN", "WERN", "WEST", "WEYS", "WFC", "WFCF", "WFRD",
    "WHF", "WILC", "WINA", "WING", "WIX",
    "WLDN", "WM", "WMB", "WMG", "WMT",
    "WOOF", "WRB", "WRLD", "WSBC", "WSBF", "WSC", "WSFS", "WSM",
    "WST", "WTBA", "WTFC", "WTW", "WULF", "WVE", "WWD", "WY", "WYNN",
    "XEL", "XENE", "XERS", "XOM", "XOMA", "XP", "XPEL", "XRAY", "XRX",
    "XYL", "YMT", "YUM", "ZBH", "ZBRA", "ZD",
    "ZION", "ZM", "ZNTL", "ZS", "ZTS", "ZUMZ", "ZURA", "ZVRA",
    # ETFs for index coverage
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
]
# Deduplicate while preserving order
TICKERS = list(dict.fromkeys(TICKERS))
N_ASSETS = len(TICKERS)
N_DAYS = 504            # ~2 trading years
SNAPSHOT_EVERY = 4      # option snapshots every 4 days
LOOKBACK = 60           # price history provided to signal generator

STRIKES_PCT = np.array(
    [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 1.00, 1.03, 1.05, 1.07, 1.10, 1.15, 1.20]
)
EXPIRIES_DAYS = np.array([7, 14, 30, 60, 90, 180])
RISK_FREE_RATE = 0.045

# Trading 212 CFD costs
CFD_SPREAD_PCT = 0.0010     # 0.10% one-way
CFD_OVERNIGHT_PCT = 0.00008 # per day (~2.9% annualized)
HOLD_DAYS = 5               # hold between snapshots
INITIAL_CAPITAL = 10_000.0
MAX_POSITION_PCT = 0.10     # max 10% of capital per asset

# ─── Black-Scholes (internal, for data generation) ───────────────────────────

def _bs_price(S, K, T, r, sigma, is_call):
    """Vectorized Black-Scholes European option pricing."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(is_call, call_price, put_price)


def _true_vol(log_moneyness, T_years, params):
    """
    Ground-truth implied volatility surface per asset.

        σ(m, T) = σ₀ + skew · ln(K/S) / √T + smile · ln(K/S)² + term · (√T − 0.3)

    Each asset has unique (σ₀, skew, smile, term) parameters derived from
    its actual realized volatility, producing realistic equity-like vol
    surfaces with negative skew and convex smile.
    """
    sigma0, skew, smile, term = params
    sqrt_T = np.sqrt(np.maximum(T_years, 1.0 / 252))
    vol = sigma0 + skew * log_moneyness / sqrt_T + smile * log_moneyness ** 2 + term * (sqrt_T - 0.3)
    return np.clip(vol, 0.05, 2.0)

# ─── Real Data Download ──────────────────────────────────────────────────────

def _download_real_prices():
    """
    Download ~2 years of daily close prices for all tickers from Yahoo Finance.
    Handles missing data gracefully for large ticker lists.
    Returns (n_valid_assets, N_DAYS) array of prices, updates TICKERS/N_ASSETS.
    """
    global TICKERS, N_ASSETS
    import yfinance as yf
    import pandas as pd

    print(f"Downloading {N_DAYS} trading days of price data for {N_ASSETS} stocks...")
    # Download in batches for reliability
    batch_size = 500
    all_dfs = []
    for i in range(0, len(TICKERS), batch_size):
        batch = TICKERS[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}: {len(batch)} tickers...")
        raw = yf.download(
            batch, period="3y", auto_adjust=True, progress=False, threads=True,
        )
        if hasattr(raw.columns, 'levels') or isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"]
        else:
            close = raw
        all_dfs.append(close)

    close = pd.concat(all_dfs, axis=1)

    # Forward-fill small gaps, then drop tickers with too many NaNs
    close = close.ffill().bfill()
    # Keep only tickers with at least N_DAYS of data
    valid_cols = close.columns[close.count() >= N_DAYS]
    close = close[valid_cols]
    close = close.iloc[-N_DAYS:]
    close = close.dropna(axis=1)  # drop any remaining NaN columns

    # Update global TICKERS to only valid ones
    valid_tickers = [t for t in TICKERS if t in close.columns]
    print(f"  {len(valid_tickers)} tickers have full price history (of {N_ASSETS} requested)")
    TICKERS = valid_tickers
    N_ASSETS = len(TICKERS)

    # Convert to numpy array (N_ASSETS, N_DAYS)
    prices = np.zeros((N_ASSETS, N_DAYS))
    for i, ticker in enumerate(TICKERS):
        prices[i] = close[ticker].values

    print(f"Got {N_DAYS} trading days from {close.index[0].date()} to {close.index[-1].date()}")
    return prices


# ─── Data Generation ─────────────────────────────────────────────────────────

def _generate_dataset():
    """
    Generate market dataset using REAL stock prices from Yahoo Finance
    with synthetic options generated on those real prices.

    Returns dict with same structure as original synthetic version.
    """
    rng = np.random.default_rng(SEED)

    # ── 1. Get real underlying prices ──
    prices = _download_real_prices()

    # ── 2. Vol surface parameters per asset (derived from real realized vol) ──
    vol_params = np.zeros((N_ASSETS, 4))
    for i in range(N_ASSETS):
        log_ret = np.diff(np.log(prices[i]))
        rv = np.std(log_ret) * np.sqrt(252)  # realized vol

        # Derive vol surface params from realized vol
        # σ₀ slightly above RV (IV typically trades at premium to RV)
        vol_params[i] = [
            rv * rng.uniform(1.0, 1.3),           # σ₀: base IV (at or above realized)
            rng.uniform(-0.20, -0.05),              # skew: negative (equity-like)
            rng.uniform(0.05, 0.25),                # smile: convex
            rng.uniform(-0.03, 0.03),               # term structure tilt
        ]

    print(f"Generated vol params for {N_ASSETS} assets (showing first 20):")
    for i, t in enumerate(TICKERS[:20]):
        rv = np.std(np.diff(np.log(prices[i]))) * np.sqrt(252)
        print(f"  {t:5s}: RV={rv:.3f}, s0={vol_params[i,0]:.3f}, "
              f"skew={vol_params[i,1]:.3f}, smile={vol_params[i,2]:.3f}, "
              f"term={vol_params[i,3]:.3f}")

    # ── 3. Generate option snapshots (vectorized per asset) ──
    snapshot_days = np.arange(LOOKBACK, N_DAYS - HOLD_DAYS, SNAPSHOT_EVERY)
    n_snaps = len(snapshot_days)
    n_strikes = len(STRIKES_PCT)
    n_expiries = len(EXPIRIES_DAYS)
    n_per_snap = n_strikes * n_expiries * 2  # calls + puts
    n_total = N_ASSETS * n_snaps * n_per_snap

    # Pre-build strike/expiry/call grids (same for every snapshot)
    strike_grid = np.tile(np.repeat(STRIKES_PCT, n_expiries * 2), n_snaps)
    expiry_grid = np.tile(np.tile(np.repeat(EXPIRIES_DAYS, 2), n_strikes), n_snaps)
    call_grid = np.tile(np.tile([True, False], n_strikes * n_expiries), n_snaps)
    T_grid = expiry_grid / 252.0
    snap_grid = np.repeat(np.arange(n_snaps), n_per_snap)

    opts_per_asset = n_snaps * n_per_snap

    opt_asset = np.zeros(n_total, dtype=np.int32)
    opt_snap = np.zeros(n_total, dtype=np.int32)
    opt_S = np.zeros(n_total)
    opt_K = np.zeros(n_total)
    opt_T = np.zeros(n_total)
    opt_is_call = np.zeros(n_total, dtype=bool)
    opt_true_iv = np.zeros(n_total)
    opt_true_price = np.zeros(n_total)
    opt_market_price = np.zeros(n_total)

    print(f"Generating {n_total:,} options for {N_ASSETS} assets...")

    for ai in range(N_ASSETS):
        if ai % 200 == 0:
            print(f"  Asset {ai}/{N_ASSETS}...")
        params = vol_params[ai]
        start = ai * opts_per_asset
        end = start + opts_per_asset

        # Spot prices for each snapshot
        spot_per_snap = prices[ai, snapshot_days]  # (n_snaps,)
        S_vec = np.repeat(spot_per_snap, n_per_snap)
        K_vec = S_vec * strike_grid
        log_m = np.log(strike_grid)  # log(K/S) = log(strike_pct)

        iv_vec = _true_vol(log_m, T_grid, params)
        tp_vec = _bs_price(S_vec, K_vec, T_grid, RISK_FREE_RATE, iv_vec, call_grid)
        tp_vec = np.maximum(tp_vec, 0.01)

        noise_pct = rng.uniform(0.005, 0.03, size=opts_per_asset)
        mp_vec = tp_vec * (1.0 + noise_pct * rng.standard_normal(opts_per_asset))
        mp_vec = np.maximum(mp_vec, 0.01)

        opt_asset[start:end] = ai
        opt_snap[start:end] = snap_grid
        opt_S[start:end] = S_vec
        opt_K[start:end] = K_vec
        opt_T[start:end] = T_grid
        opt_is_call[start:end] = call_grid
        opt_true_iv[start:end] = iv_vec
        opt_true_price[start:end] = tp_vec
        opt_market_price[start:end] = mp_vec

    return {
        "prices": prices,
        "vol_params": vol_params,
        "snapshot_days": snapshot_days,
        "opt_asset": opt_asset,
        "opt_snap": opt_snap,
        "opt_S": opt_S,
        "opt_K": opt_K,
        "opt_T": opt_T,
        "opt_is_call": opt_is_call,
        "opt_true_iv": opt_true_iv,
        "opt_true_price": opt_true_price,
        "opt_market_price": opt_market_price,
    }


def load_data():
    """Load cached dataset or generate fresh."""
    if DATA_FILE.exists():
        data = dict(np.load(DATA_FILE, allow_pickle=False))
        # Restore bool dtype
        data["opt_is_call"] = data["opt_is_call"].astype(bool)
        return data

    print("Generating real-data market dataset (first run only)...")
    t0 = time.time()
    data = _generate_dataset()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_FILE, **data)
    print(f"Dataset cached to {DATA_FILE} ({time.time() - t0:.1f}s)")
    return data

# ─── CFD Trade Simulation ────────────────────────────────────────────────────

def simulate_cfd_trades(signals, prices, snapshot_days):
    """
    Simulate Trading 212-style CFD trades from model signals.

    Args:
        signals:       (N_ASSETS, n_snapshots) float in [-1, 1]
                       positive = long, negative = short, 0 = flat
        prices:        (N_ASSETS, N_DAYS) underlying daily closes
        snapshot_days: (n_snapshots,) day indices

    Returns:
        dict with: sharpe, total_return, max_drawdown, win_rate, n_trades,
                   daily_pnl (array)
    """
    n_assets, n_snaps = signals.shape
    capital = INITIAL_CAPITAL
    daily_equity = [capital]
    trade_returns = []

    for si in range(n_snaps):
        day = snapshot_days[si]
        exit_day = min(day + HOLD_DAYS, prices.shape[1] - 1)
        if exit_day <= day:
            continue

        snap_pnl = 0.0
        for ai in range(n_assets):
            sig = signals[ai, si]
            if abs(sig) < 0.05:  # dead zone
                continue

            entry_price = prices[ai, day]
            exit_price = prices[ai, exit_day]
            if entry_price <= 0:
                continue

            # Position size: signal strength × max allocation
            pos_size = abs(sig) * MAX_POSITION_PCT * capital
            direction = np.sign(sig)

            # Entry cost (half-spread)
            effective_entry = entry_price * (1.0 + direction * CFD_SPREAD_PCT)
            effective_exit = exit_price * (1.0 - direction * CFD_SPREAD_PCT)

            # Gross return
            gross_ret = direction * (effective_exit - effective_entry) / entry_price

            # Overnight financing
            hold = exit_day - day
            financing = CFD_OVERNIGHT_PCT * hold

            net_ret = gross_ret - financing
            pnl = pos_size * net_ret
            snap_pnl += pnl
            trade_returns.append(net_ret)

        capital += snap_pnl
        capital = max(capital, 1.0)  # floor to prevent zero
        daily_equity.append(capital)

    daily_equity = np.array(daily_equity)
    equity_returns = np.diff(daily_equity) / daily_equity[:-1]

    # Metrics
    trade_returns = np.array(trade_returns) if trade_returns else np.array([0.0])
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Sharpe (annualized, using snapshot-frequency returns)
    if len(equity_returns) > 1 and equity_returns.std() > 1e-10:
        periods_per_year = 252 / HOLD_DAYS
        sharpe = equity_returns.mean() / equity_returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(daily_equity)
    drawdown = (peak - daily_equity) / peak
    max_dd = drawdown.max()

    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "n_trades": len(trade_returns),
        "daily_equity": daily_equity,
    }

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model):
    """
    Full evaluation pipeline — the single metric the agent optimizes.

    The model must implement:
        model.price_chain(chain: dict) -> np.ndarray of fair values
        model.generate_signal(chain: dict, price_history: np.ndarray) -> float in [-1, 1]

    Prints summary to stdout (agent greps for combined_score).
    Returns the combined score (higher is better).
    """
    t0 = time.time()
    data = load_data()

    prices = data["prices"]
    snapshot_days = data["snapshot_days"]
    n_snaps = len(snapshot_days)

    # Pre-build group index: (asset, snapshot) -> array of option indices
    # This avoids repeated O(n) mask lookups in the inner loop.
    group_idx = {}
    for i in range(len(data["opt_asset"])):
        key = (int(data["opt_asset"][i]), int(data["opt_snap"][i]))
        if key not in group_idx:
            group_idx[key] = []
        group_idx[key].append(i)
    group_idx = {k: np.array(v) for k, v in group_idx.items()}

    def _make_chain(idx):
        return {
            "S": data["opt_S"][idx],
            "K": data["opt_K"][idx],
            "T": data["opt_T"][idx],
            "r": RISK_FREE_RATE,
            "is_call": data["opt_is_call"][idx],
            "market_price": data["opt_market_price"][idx],
        }

    # ── 1. Pricing accuracy + 2. Signal generation (parallelized per asset) ──
    model_prices = np.copy(data["opt_market_price"])  # fallback = market price
    signals = np.zeros((N_ASSETS, n_snaps))

    def _process_asset(ai):
        """Process pricing + signals for a single asset."""
        local_prices = {}
        local_signals = np.zeros(n_snaps)
        for si, day in enumerate(snapshot_days):
            key = (ai, si)
            if key not in group_idx:
                continue
            idx = group_idx[key]
            chain = _make_chain(idx)

            # Pricing
            try:
                fv = model.price_chain(chain)
                local_prices[si] = (idx, np.asarray(fv).flatten())
            except Exception:
                pass

            # Signal
            history_start = max(0, day - LOOKBACK)
            price_history = prices[ai, history_start:day + 1]
            try:
                sig = model.generate_signal(chain, price_history)
                local_signals[si] = float(np.clip(sig, -1.0, 1.0))
            except Exception:
                local_signals[si] = 0.0

        return ai, local_prices, local_signals

    # Run in parallel across assets
    n_jobs = min(8, max(1, N_ASSETS // 10))  # scale workers with asset count
    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_process_asset)(ai) for ai in range(N_ASSETS)
    )

    for ai, local_prices, local_signals in results:
        for si, (idx, fv) in local_prices.items():
            model_prices[idx] = fv
        signals[ai] = local_signals

    # MAPE against TRUE prices (not market prices)
    true_p = data["opt_true_price"]
    mape = np.mean(np.abs(model_prices - true_p) / np.maximum(true_p, 0.01))

    # RMSE normalized by mean price
    rmse = np.sqrt(np.mean((model_prices - true_p) ** 2))
    rmse_pct = rmse / np.mean(true_p)

    # ── 3. CFD simulation ──
    cfd = simulate_cfd_trades(signals, prices, snapshot_days)

    # ── 4. Combined score ──
    # Higher is better. Pricing accuracy is rewarded (lower MAPE = higher score),
    # signal profitability is rewarded (higher Sharpe = higher score).
    # Weighting: 40% pricing, 60% signals (because the user trades CFDs).
    pricing_score = max(0, 1.0 - mape)  # 1.0 = perfect, 0.0 = 100% MAPE
    signal_score = cfd["sharpe"]         # typically -2 to +3

    combined = 0.4 * pricing_score + 0.6 * signal_score

    elapsed = time.time() - t0

    # ── Print results (agent greps these) ──
    print(f"combined_score {combined:.6f}")
    print(f"pricing_mape {mape:.6f}")
    print(f"pricing_rmse_pct {rmse_pct:.6f}")
    print(f"sharpe_ratio {cfd['sharpe']:.6f}")
    print(f"total_return {cfd['total_return']:.6f}")
    print(f"max_drawdown {cfd['max_drawdown']:.6f}")
    print(f"win_rate {cfd['win_rate']:.6f}")
    print(f"n_trades {cfd['n_trades']}")
    print(f"eval_time {elapsed:.1f}s")

    return combined
