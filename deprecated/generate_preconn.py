"""
Comprehensive Linguistic Reasoning Dataset Generator
Creates sophisticated puzzles with wordplay, hidden connections, and misdirection
Designed for maximum linguistic complexity and reasoning challenges
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Comprehensive linguistic pattern examples with subgroup tracking
LINGUISTIC_PATTERNS = {
    "PHONETIC_PATTERNS": {
        "description": "Words related by sound patterns and pronunciation",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Homophones (same sound)", "words": ["KNIGHT", "NIGHT", "WRITE", "RIGHT", "RITE", "WRIGHT", "SITE", "SIGHT", "CITE", "BYTE", "BITE", "BIGHT"]},
            {"subgroup": "Near homophones", "words": ["ACCEPT", "EXCEPT", "AFFECT", "EFFECT", "ALLUSION", "ILLUSION", "ELICIT", "ILLICIT", "EMIGRATE", "IMMIGRATE"]},
            {"subgroup": "Silent letter patterns", "words": ["KNIFE", "KNOW", "KNEE", "KNOT", "GNOME", "GNAT", "GNAW", "PSALM", "PSYCHOLOGY", "PNEUMONIA", "PTERODACTYL"]},
            {"subgroup": "Rhyme families (-ATE)", "words": ["CREATE", "DEBATE", "ESTATE", "RELATE", "SEDATE", "UPDATE", "VIBRATE", "MIGRATE", "DONATE", "ROTATE", "LOCATE", "EDUCATE"]},
            {"subgroup": "Rhyme families (-TION)", "words": ["NATION", "STATION", "CREATION", "VACATION", "EDUCATION", "FOUNDATION", "CELEBRATION", "INNOVATION", "MEDITATION", "RADIATION"]},
            {"subgroup": "Alliterative sets", "words": ["BIGGER", "BETTER", "BITTER", "BUTTER", "BATTER", "BANNER", "BORDER", "BROTHER", "BOULDER", "BUILDER", "BUMPER", "BUNKER"]},
            {"subgroup": "Assonance patterns", "words": ["FLEET", "SLEEP", "DEEP", "KEEP", "SHEEP", "STEEP", "SWEEP", "CREEP", "CHEAP", "HEAP", "LEAP", "REAP"]},
            {"subgroup": "Consonance patterns", "words": ["LUCK", "LOOK", "LAKE", "LIKE", "LOCK", "LEAK", "LACK", "LICK", "LURK", "LINK", "LANK", "LUNK"]}
        ]
    },
    
    "MORPHOLOGICAL_PATTERNS": {
        "description": "Words related by structure and formation",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Prefix UN-", "words": ["UNABLE", "UNCERTAIN", "UNCLEAR", "UNDO", "UNFAIR", "UNHAPPY", "UNKNOWN", "UNLOCK", "UNSAFE", "UNUSUAL", "UNWRAP", "UNZIP"]},
            {"subgroup": "Prefix RE-", "words": ["REBUILD", "RECYCLE", "REDEFINE", "REFILL", "REGAIN", "REHEAT", "REJOICE", "REKINDLE", "RELOAD", "REMAKE", "RENEW", "REOPEN"]},
            {"subgroup": "Suffix -NESS", "words": ["DARKNESS", "KINDNESS", "MADNESS", "SADNESS", "WEAKNESS", "ILLNESS", "WELLNESS", "BOLDNESS", "COLDNESS", "RICHNESS", "THICKNESS"]},
            {"subgroup": "Suffix -ABLE/-IBLE", "words": ["READABLE", "VISIBLE", "POSSIBLE", "TERRIBLE", "HORRIBLE", "CAPABLE", "SUITABLE", "VALUABLE", "RELIABLE", "FLEXIBLE", "SENSIBLE"]},
            {"subgroup": "Compound words (FIRE+)", "words": ["FIREPLACE", "FIREWORK", "FIREFLY", "FIREFIGHTER", "FIREARM", "FIREWALL", "FIREPROOF", "FIRESTORM", "FIREBALL", "FIREBRAND"]},
            {"subgroup": "Compound words (+HOUSE)", "words": ["GREENHOUSE", "WAREHOUSE", "COURTHOUSE", "LIGHTHOUSE", "FARMHOUSE", "TOWNHOUSE", "DOGHOUSE", "BIRDHOUSE", "TREEHOUSE", "BOATHOUSE"]},
            {"subgroup": "Portmanteaus", "words": ["BRUNCH", "SMOG", "MOTEL", "BREXIT", "BLOG", "PODCAST", "WEBINAR", "EMOTICON", "MALWARE", "SPORK", "GLAMPING", "STAYCATION"]},
            {"subgroup": "Contractions", "words": ["CAN'T", "WON'T", "SHOULDN'T", "WOULDN'T", "COULDN'T", "DIDN'T", "HASN'T", "HAVEN'T", "ISN'T", "AREN'T", "WASN'T", "WEREN'T"]}
        ]
    },
    
    "SEMANTIC_RELATIONSHIPS": {
        "description": "Words connected by meaning and usage",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Synonyms for SMART", "words": ["INTELLIGENT", "CLEVER", "BRILLIANT", "BRIGHT", "SHARP", "QUICK", "ASTUTE", "SHREWD", "WISE", "GENIUS", "GIFTED", "BRAINY"]},
            {"subgroup": "Antonym pairs", "words": ["HOT", "COLD", "BIG", "SMALL", "FAST", "SLOW", "HIGH", "LOW", "DARK", "LIGHT", "HARD", "SOFT", "THICK", "THIN"]},
            {"subgroup": "Hypernyms (categories)", "words": ["FURNITURE", "VEHICLE", "ANIMAL", "PLANT", "FOOD", "CLOTHING", "TOOL", "WEAPON", "INSTRUMENT", "CONTAINER", "BUILDING"]},
            {"subgroup": "Meronyms (parts)", "words": ["WHEEL", "ENGINE", "DOOR", "WINDOW", "ROOF", "WALL", "FLOOR", "CEILING", "HANDLE", "BLADE", "SCREEN", "KEYBOARD"]},
            {"subgroup": "Collocations with MAKE", "words": ["DECISION", "MISTAKE", "MONEY", "SENSE", "DIFFERENCE", "PROGRESS", "EFFORT", "CHOICE", "CHANGE", "PROMISE", "PLAN", "DEAL"]},
            {"subgroup": "Collocations with HEAVY", "words": ["RAIN", "TRAFFIC", "LOAD", "BURDEN", "HEART", "METAL", "INDUSTRY", "MACHINERY", "LIFTING", "BREATHING", "SMOKER", "DRINKER"]},
            {"subgroup": "Semantic fields (CRIME)", "words": ["THEFT", "MURDER", "FRAUD", "ASSAULT", "ROBBERY", "BURGLARY", "ARSON", "KIDNAPPING", "SMUGGLING", "FORGERY", "BLACKMAIL"]},
            {"subgroup": "Gradable scales", "words": ["FREEZING", "COLD", "COOL", "LUKEWARM", "WARM", "HOT", "BOILING", "SCORCHING", "BLAZING", "INFERNAL", "VOLCANIC"]}
        ]
    },
    
    "ORTHOGRAPHIC_PATTERNS": {
        "description": "Words related by spelling patterns",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Palindromes", "words": ["LEVEL", "RADAR", "ROTOR", "CIVIC", "KAYAK", "REFER", "MADAM", "STATS", "TENET", "NOON", "RACECAR", "ROTATOR"]},
            {"subgroup": "Double letters", "words": ["BALLOON", "COFFEE", "MIDDLE", "PEPPER", "YELLOW", "PUZZLE", "BUBBLE", "SADDLE", "BATTLE", "BOTTLE", "SETTLE", "LITTLE"]},
            {"subgroup": "Q without U", "words": ["QI", "QOPH", "QADI", "QAID", "QANAT", "QAT", "QAWWALI", "QIGONG", "QINDAR", "QINTAR", "QWERTY", "FAQIR"]},
            {"subgroup": "All vowels", "words": ["SEQUOIA", "AUREOLE", "EULOGIA", "AERIOUS", "AQUEOUS", "CAESIOUS", "EUCOSIA", "EUTOPIA", "MIAOUED", "QUEUE", "AUDIO"]},
            {"subgroup": "No vowels (Y as vowel)", "words": ["RHYTHM", "MYTH", "LYNX", "GYPSY", "CRYPT", "TRYST", "NYMPH", "PSYCH", "SYLPH", "LYNCH", "SYNTH", "GLYPH"]},
            {"subgroup": "Consecutive alphabetical", "words": ["FIRST", "ALMOST", "BIOPSY", "CHINTZ", "GHOSTLY", "ABHORS", "BEGINS", "CHIMP", "DEFROST", "HIJACK"]},
            {"subgroup": "Anagram sets", "words": ["LISTEN", "SILENT", "ENLIST", "EARTH", "HEART", "HATER", "ANGEL", "ANGLE", "GLEAN", "SPARE", "SPEAR", "PARSE"]},
            {"subgroup": "Letter patterns (ABAB)", "words": ["PAPA", "MAMA", "NANA", "DODO", "TOTO", "LULU", "MIMI", "COCO", "GAGA", "KIKI", "JUJU", "BOBO"]}
        ]
    },
    
    "ETYMOLOGICAL_PATTERNS": {
        "description": "Words grouped by origin and history",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Latin roots (AQUA)", "words": ["AQUARIUM", "AQUATIC", "AQUEDUCT", "AQUIFER", "AQUEOUS", "SUBAQUEOUS", "AQUACULTURE", "AQUAMARINE", "AQUAPLANE"]},
            {"subgroup": "Greek roots (PHOTO)", "words": ["PHOTOGRAPH", "PHOTOSYNTHESIS", "PHOTOCOPY", "PHOTOGENIC", "PHOTOMETRY", "PHOTOSPHERE", "PHOTOTROPISM", "PHOTOPHOBIA"]},
            {"subgroup": "French loanwords", "words": ["RESTAURANT", "BOUTIQUE", "LINGERIE", "CUISINE", "BALLET", "CHAMPAGNE", "CHAUFFEUR", "CLICHÉ", "DÉBACLE", "FAÇADE"]},
            {"subgroup": "Germanic roots", "words": ["HOUSE", "BREAD", "WATER", "EARTH", "FIRE", "STONE", "WOOD", "IRON", "GOLD", "SILVER", "COPPER", "BRONZE"]},
            {"subgroup": "Arabic loanwords", "words": ["ALGEBRA", "ALGORITHM", "ALCOHOL", "COFFEE", "COTTON", "MAGAZINE", "SAFARI", "SULTAN", "ZENITH", "ZERO", "CIPHER"]},
            {"subgroup": "Japanese loanwords", "words": ["KARAOKE", "TSUNAMI", "SUSHI", "ORIGAMI", "KARATE", "NINJA", "SAMURAI", "EMOJI", "MANGA", "ANIME", "SUDOKU", "TERIYAKI"]},
            {"subgroup": "Italian loanwords", "words": ["PIANO", "OPERA", "SOPRANO", "PASTA", "PIZZA", "ESPRESSO", "CAPPUCCINO", "GRAFFITI", "PAPARAZZI", "SCENARIO", "FIASCO"]},
            {"subgroup": "Sanskrit roots", "words": ["YOGA", "KARMA", "MANTRA", "GURU", "AVATAR", "NIRVANA", "PUNDIT", "JUNGLE", "SHAMPOO", "CANDY", "BANDANA", "CHEETAH"]}
        ]
    },
    
    "CATEGORICAL_PATTERNS": {
        "description": "Words belonging to specific categories",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Zodiac signs", "words": ["ARIES", "TAURUS", "GEMINI", "CANCER", "LEO", "VIRGO", "LIBRA", "SCORPIO", "SAGITTARIUS", "CAPRICORN", "AQUARIUS", "PISCES"]},
            {"subgroup": "Planets", "words": ["MERCURY", "VENUS", "EARTH", "MARS", "JUPITER", "SATURN", "URANUS", "NEPTUNE", "PLUTO", "CERES", "ERIS", "MAKEMAKE"]},
            {"subgroup": "Chess pieces", "words": ["KING", "QUEEN", "ROOK", "BISHOP", "KNIGHT", "PAWN", "CASTLE", "HORSE"]},
            {"subgroup": "Card suits", "words": ["HEARTS", "DIAMONDS", "CLUBS", "SPADES", "TRUMP", "JOKER", "ACE", "KING", "QUEEN", "JACK"]},
            {"subgroup": "Musical notes", "words": ["WHOLE", "HALF", "QUARTER", "EIGHTH", "SIXTEENTH", "SHARP", "FLAT", "NATURAL", "REST", "FERMATA", "STACCATO", "LEGATO"]},
            {"subgroup": "Punctuation names", "words": ["PERIOD", "COMMA", "SEMICOLON", "COLON", "APOSTROPHE", "QUOTATION", "EXCLAMATION", "QUESTION", "HYPHEN", "DASH", "PARENTHESIS"]},
            {"subgroup": "Mathematical operations", "words": ["ADDITION", "SUBTRACTION", "MULTIPLICATION", "DIVISION", "EXPONENT", "ROOT", "FACTORIAL", "LOGARITHM", "INTEGRAL", "DERIVATIVE"]},
            {"subgroup": "Programming concepts", "words": ["VARIABLE", "FUNCTION", "LOOP", "ARRAY", "OBJECT", "CLASS", "METHOD", "PARAMETER", "ARGUMENT", "RETURN", "EXCEPTION", "INTERFACE"]}
        ]
    },
    
    "CONTEXTUAL_PATTERNS": {
        "description": "Words that change meaning based on context",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Heteronyms (different pronunciation)", "words": ["LEAD", "READ", "TEAR", "BASS", "BOW", "CLOSE", "DESERT", "INVALID", "OBJECT", "PRESENT", "PRODUCE", "REFUSE"]},
            {"subgroup": "Contronyms (self-antonyms)", "words": ["CLEAVE", "DUST", "FAST", "LEFT", "OVERSIGHT", "SANCTION", "SCREEN", "SEED", "STRIKE", "TRIM", "WEATHER", "WIND"]},
            {"subgroup": "Polysemes (multiple meanings)", "words": ["BANK", "BARK", "BAT", "BEAR", "BOWL", "CRANE", "CURRENT", "DATE", "FAIR", "FILE", "FIRM", "MATCH"]},
            {"subgroup": "Homographs (same spelling)", "words": ["MINUTE", "POLISH", "RECORD", "PERMIT", "CONTEST", "CONDUCT", "EXTRACT", "IMPORT", "INSULT", "PERFECT", "PROJECT"]},
            {"subgroup": "Phrasal verb particles", "words": ["UP", "DOWN", "IN", "OUT", "ON", "OFF", "OVER", "UNDER", "THROUGH", "AROUND", "ALONG", "ACROSS"]},
            {"subgroup": "Modal verbs", "words": ["CAN", "COULD", "MAY", "MIGHT", "MUST", "SHALL", "SHOULD", "WILL", "WOULD", "OUGHT", "NEED", "DARE"]},
            {"subgroup": "Discourse markers", "words": ["HOWEVER", "THEREFORE", "MOREOVER", "FURTHERMORE", "NEVERTHELESS", "NONETHELESS", "CONSEQUENTLY", "MEANWHILE", "OTHERWISE", "INDEED"]},
            {"subgroup": "Hedge words", "words": ["PERHAPS", "MAYBE", "POSSIBLY", "PROBABLY", "PRESUMABLY", "APPARENTLY", "SEEMINGLY", "RELATIVELY", "SOMEWHAT", "FAIRLY", "QUITE"]}
        ]
    },
    
    "COGNITIVE_PATTERNS": {
        "description": "Words related to mental processes and concepts",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Emotions", "words": ["HAPPY", "SAD", "ANGRY", "AFRAID", "SURPRISED", "DISGUSTED", "ANXIOUS", "EXCITED", "CALM", "JEALOUS", "PROUD", "ASHAMED"]},
            {"subgroup": "Thinking verbs", "words": ["THINK", "KNOW", "BELIEVE", "UNDERSTAND", "REMEMBER", "FORGET", "IMAGINE", "WONDER", "SUPPOSE", "ASSUME", "CONCLUDE"]},
            {"subgroup": "Perception verbs", "words": ["SEE", "HEAR", "FEEL", "SMELL", "TASTE", "SENSE", "NOTICE", "OBSERVE", "DETECT", "PERCEIVE", "RECOGNIZE", "IDENTIFY"]},
            {"subgroup": "Memory types", "words": ["EPISODIC", "SEMANTIC", "PROCEDURAL", "DECLARATIVE", "IMPLICIT", "EXPLICIT", "WORKING", "SENSORY", "SHORT-TERM", "LONG-TERM"]},
            {"subgroup": "Learning styles", "words": ["VISUAL", "AUDITORY", "KINESTHETIC", "READING", "WRITING", "LOGICAL", "SOCIAL", "SOLITARY", "VERBAL", "PHYSICAL", "NATURALIST"]},
            {"subgroup": "Intelligence types", "words": ["LINGUISTIC", "LOGICAL", "SPATIAL", "MUSICAL", "BODILY", "INTERPERSONAL", "INTRAPERSONAL", "NATURALISTIC", "EXISTENTIAL", "EMOTIONAL"]},
            {"subgroup": "Cognitive biases", "words": ["CONFIRMATION", "ANCHORING", "AVAILABILITY", "HINDSIGHT", "DUNNING-KRUGER", "RECENCY", "PRIMACY", "HALO", "HORN", "BANDWAGON"]},
            {"subgroup": "Logical fallacies", "words": ["STRAWMAN", "SLIPPERY-SLOPE", "CIRCULAR", "HASTY", "POST-HOC", "FALSE-DILEMMA", "BANDWAGON", "APPEAL", "COMPOSITION", "DIVISION"]}
        ]
    },
    
    "CULTURAL_PATTERNS": {
        "description": "Words from specific cultural contexts",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Shakespeare coinages", "words": ["ASSASSINATION", "BEDAZZLED", "COUNTLESS", "EYEBALL", "FASHIONABLE", "GENEROUS", "HURRIED", "LONELY", "MAJESTIC", "OBSCENE"]},
            {"subgroup": "Internet slang", "words": ["MEME", "VIRAL", "TROLL", "SPAM", "BLOG", "VLOG", "EMOJI", "HASHTAG", "SELFIE", "TWEET", "STREAM", "PODCAST"]},
            {"subgroup": "Business jargon", "words": ["SYNERGY", "LEVERAGE", "PARADIGM", "BANDWIDTH", "PIPELINE", "ECOSYSTEM", "DISRUPT", "PIVOT", "SCALE", "ITERATE", "AGILE"]},
            {"subgroup": "Legal terms", "words": ["PLAINTIFF", "DEFENDANT", "LITIGATION", "PRECEDENT", "JURISDICTION", "TESTIMONY", "VERDICT", "APPEAL", "INJUNCTION", "SUBPOENA"]},
            {"subgroup": "Medical terms", "words": ["DIAGNOSIS", "PROGNOSIS", "SYMPTOM", "SYNDROME", "CHRONIC", "ACUTE", "BENIGN", "MALIGNANT", "THERAPY", "TREATMENT", "PRESCRIPTION"]},
            {"subgroup": "Culinary terms", "words": ["SAUTÉ", "BRAISE", "POACH", "BLANCH", "JULIENNE", "DICE", "MINCE", "PURÉE", "CARAMELIZE", "DEGLAZE", "FLAMBÉ", "MARINADE"]},
            {"subgroup": "Fashion terms", "words": ["HAUTE", "COUTURE", "PRÊT-À-PORTER", "AVANT-GARDE", "VINTAGE", "CHIC", "BOHEMIAN", "MINIMALIST", "ECLECTIC", "TIMELESS"]},
            {"subgroup": "Sports terminology", "words": ["OFFENSE", "DEFENSE", "OVERTIME", "PENALTY", "REFEREE", "FOUL", "TIMEOUT", "PLAYOFFS", "CHAMPIONSHIP", "TOURNAMENT", "LEAGUE"]}
        ]
    },
    
    "LINGUISTIC_PHENOMENA": {
        "description": "Words demonstrating specific linguistic phenomena",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Onomatopoeia", "words": ["BANG", "CRASH", "SIZZLE", "BUZZ", "HISS", "CLICK", "SPLASH", "THUD", "WHOOSH", "CRACKLE", "RUSTLE", "MURMUR"]},
            {"subgroup": "Reduplication", "words": ["PING-PONG", "FLIP-FLOP", "ZIG-ZAG", "TIP-TOP", "TICK-TOCK", "CHIT-CHAT", "KNICK-KNACK", "PITTER-PATTER", "RIFF-RAFF"]},
            {"subgroup": "Metonymy examples", "words": ["CROWN", "WHITEHOUSE", "HOLLYWOOD", "WALLSTREET", "SILICON", "VALLEY", "PENTAGON", "KREMLIN", "DOWNING", "BROADWAY", "FLEET"]},
            {"subgroup": "Euphemisms", "words": ["PASSED", "RESTROOM", "EXPECTING", "CHALLENGED", "SENIOR", "PREOWNED", "DOWNSIZING", "COLLATERAL", "ENHANCED", "CORRECTIONAL"]},
            {"subgroup": "Dysphemisms", "words": ["SNAIL-MAIL", "DEAD-TREE", "IDIOT-BOX", "GAS-GUZZLER", "BEAN-COUNTER", "PENCIL-PUSHER", "SHRINK", "QUACK", "EGGHEAD"]},
            {"subgroup": "Backronyms", "words": ["SPAM", "RADAR", "LASER", "SCUBA", "YAHOO", "WIKI", "SMART", "CAPTCHA", "PATRIOT", "AMBER", "DARE", "MADD"]},
            {"subgroup": "Spoonerisms base", "words": ["BUTTERFLY", "CRUSHING", "BLOW", "WELL-OILED", "BICYCLE", "PACK", "LIES", "NOSEY", "COOK", "WAVE", "FLAG", "SHOWING"]},
            {"subgroup": "Malapropisms source", "words": ["ILLITERATE", "ALLEGORY", "PINEAPPLE", "EPITAPH", "CONTAGIOUS", "PERPENDICULAR", "SUPERFICIAL", "PARTICIPATE", "FLAMENCO"]}
        ]
    },
    
    "COMPUTATIONAL_PATTERNS": {
        "description": "Words from computing and technology",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Programming keywords", "words": ["IF", "ELSE", "WHILE", "FOR", "RETURN", "BREAK", "CONTINUE", "SWITCH", "CASE", "TRY", "CATCH", "FINALLY"]},
            {"subgroup": "Data structures", "words": ["ARRAY", "LIST", "STACK", "QUEUE", "TREE", "GRAPH", "HEAP", "HASH", "TABLE", "MATRIX", "VECTOR", "SET"]},
            {"subgroup": "Algorithms", "words": ["SORT", "SEARCH", "MERGE", "QUICK", "BUBBLE", "HEAP", "BINARY", "LINEAR", "DEPTH", "BREADTH", "DIJKSTRA", "DYNAMIC"]},
            {"subgroup": "Network terms", "words": ["ROUTER", "SWITCH", "GATEWAY", "FIREWALL", "PROXY", "SERVER", "CLIENT", "PROTOCOL", "PACKET", "BANDWIDTH", "LATENCY"]},
            {"subgroup": "Database terms", "words": ["SELECT", "INSERT", "UPDATE", "DELETE", "JOIN", "WHERE", "GROUP", "ORDER", "INDEX", "PRIMARY", "FOREIGN", "CONSTRAINT"]},
            {"subgroup": "Web technologies", "words": ["HTML", "CSS", "JAVASCRIPT", "AJAX", "JSON", "XML", "API", "REST", "SOAP", "COOKIE", "SESSION", "CACHE"]},
            {"subgroup": "Security terms", "words": ["ENCRYPTION", "HASH", "SALT", "TOKEN", "CERTIFICATE", "AUTHENTICATION", "AUTHORIZATION", "FIREWALL", "MALWARE", "PHISHING"]},
            {"subgroup": "AI/ML terms", "words": ["NEURAL", "NETWORK", "DEEP", "LEARNING", "TRAINING", "MODEL", "DATASET", "FEATURE", "LABEL", "PREDICTION", "CLASSIFICATION"]}
        ]
    },
    
    "TEMPORAL_PATTERNS": {
        "description": "Words related to time and sequence",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Days of week", "words": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY", "WEEKDAY", "WEEKEND", "TODAY", "TOMORROW"]},
            {"subgroup": "Months", "words": ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]},
            {"subgroup": "Seasons", "words": ["SPRING", "SUMMER", "AUTUMN", "FALL", "WINTER", "SOLSTICE", "EQUINOX", "SEASONAL", "MONSOON", "HARVEST", "PLANTING"]},
            {"subgroup": "Time periods", "words": ["SECOND", "MINUTE", "HOUR", "DAY", "WEEK", "MONTH", "YEAR", "DECADE", "CENTURY", "MILLENNIUM", "EPOCH", "ERA"]},
            {"subgroup": "Frequency adverbs", "words": ["ALWAYS", "USUALLY", "OFTEN", "SOMETIMES", "OCCASIONALLY", "RARELY", "SELDOM", "NEVER", "DAILY", "WEEKLY", "MONTHLY", "YEARLY"]},
            {"subgroup": "Sequence markers", "words": ["FIRST", "SECOND", "THIRD", "NEXT", "THEN", "AFTER", "BEFORE", "DURING", "WHILE", "MEANWHILE", "FINALLY", "LASTLY"]},
            {"subgroup": "Historical periods", "words": ["ANCIENT", "CLASSICAL", "MEDIEVAL", "RENAISSANCE", "BAROQUE", "ENLIGHTENMENT", "INDUSTRIAL", "MODERN", "CONTEMPORARY", "POSTMODERN"]},
            {"subgroup": "Geological time", "words": ["PRECAMBRIAN", "PALEOZOIC", "MESOZOIC", "CENOZOIC", "JURASSIC", "CRETACEOUS", "TRIASSIC", "PERMIAN", "DEVONIAN", "CAMBRIAN"]}
        ]
    },
    
    "SPATIAL_PATTERNS": {
        "description": "Words related to space and location",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Cardinal directions", "words": ["NORTH", "SOUTH", "EAST", "WEST", "NORTHEAST", "NORTHWEST", "SOUTHEAST", "SOUTHWEST", "CENTRAL", "MIDDLE"]},
            {"subgroup": "Relative positions", "words": ["ABOVE", "BELOW", "BESIDE", "BETWEEN", "AMONG", "BEHIND", "FRONT", "BACK", "LEFT", "RIGHT", "CENTER", "EDGE"]},
            {"subgroup": "Prepositions of place", "words": ["IN", "ON", "AT", "UNDER", "OVER", "THROUGH", "ACROSS", "ALONG", "AROUND", "AGAINST", "TOWARD", "BEYOND"]},
            {"subgroup": "Geographic features", "words": ["MOUNTAIN", "VALLEY", "RIVER", "OCEAN", "LAKE", "FOREST", "DESERT", "PLAIN", "PLATEAU", "CANYON", "PENINSULA", "ISLAND"]},
            {"subgroup": "Urban features", "words": ["STREET", "AVENUE", "BOULEVARD", "ROAD", "HIGHWAY", "BRIDGE", "TUNNEL", "INTERSECTION", "ROUNDABOUT", "PLAZA", "SQUARE"]},
            {"subgroup": "Architectural spaces", "words": ["ROOM", "HALL", "CORRIDOR", "LOBBY", "ATTIC", "BASEMENT", "BALCONY", "TERRACE", "COURTYARD", "FOYER", "VESTIBULE"]},
            {"subgroup": "Measurement units", "words": ["INCH", "FOOT", "YARD", "MILE", "METER", "KILOMETER", "CENTIMETER", "MILLIMETER", "HECTARE", "ACRE", "SQUARE", "CUBIC"]},
            {"subgroup": "Shape descriptors", "words": ["ROUND", "SQUARE", "TRIANGULAR", "RECTANGULAR", "OVAL", "CIRCULAR", "SPHERICAL", "CUBIC", "CYLINDRICAL", "CONICAL"]}
        ]
    },
    
    "REGISTER_PATTERNS": {
        "description": "Words from different language registers",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Formal register", "words": ["COMMENCE", "TERMINATE", "ACQUIRE", "ENDEAVOR", "ASCERTAIN", "FACILITATE", "IMPLEMENT", "PURSUANT", "HERETOFORE", "WHEREAS"]},
            {"subgroup": "Informal register", "words": ["STUFF", "THING", "GONNA", "WANNA", "KINDA", "SORTA", "YEAH", "NOPE", "OKAY", "COOL", "AWESOME", "TOTALLY"]},
            {"subgroup": "Academic register", "words": ["HYPOTHESIS", "METHODOLOGY", "PARADIGM", "THEORETICAL", "EMPIRICAL", "QUALITATIVE", "QUANTITATIVE", "CORRELATION", "SYNTHESIS"]},
            {"subgroup": "Technical register", "words": ["SPECIFICATION", "PARAMETER", "CONFIGURATION", "OPTIMIZATION", "CALIBRATION", "IMPLEMENTATION", "PROTOCOL", "INTERFACE"]},
            {"subgroup": "Poetic register", "words": ["WHENCE", "THITHER", "HITHER", "YONDER", "BETWIXT", "ERE", "OFT", "NEATH", "TWAS", "MIDST", "OERSHADOW", "FORSOOTH"]},
            {"subgroup": "Colloquialisms", "words": ["BUDDY", "FOLKS", "GUYS", "BUNCH", "LOADS", "TONS", "HEAPS", "PRETTY", "QUITE", "RATHER", "FAIRLY", "SOMEWHAT"]},
            {"subgroup": "Archaic terms", "words": ["THOU", "THEE", "THY", "THINE", "YE", "HATH", "DOTH", "SHALT", "WHILST", "AMONGST", "BETWIXT", "WHEREFORE"]},
            {"subgroup": "Neologisms", "words": ["CRYPTOCURRENCY", "BLOCKCHAIN", "METAVERSE", "INFLUENCER", "UNFOLLOW", "LIVESTREAM", "CROWDFUND", "MICRODOSE", "GHOSTING", "CATFISH"]}
        ]
    },
    
    "SYNTACTIC_PATTERNS": {
        "description": "Words grouped by grammatical function",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "Auxiliary verbs", "words": ["BE", "HAVE", "DO", "CAN", "COULD", "MAY", "MIGHT", "MUST", "SHALL", "SHOULD", "WILL", "WOULD"]},
            {"subgroup": "Determiners", "words": ["THE", "A", "AN", "THIS", "THAT", "THESE", "THOSE", "MY", "YOUR", "HIS", "HER", "ITS", "OUR", "THEIR"]},
            {"subgroup": "Conjunctions", "words": ["AND", "OR", "BUT", "NOR", "FOR", "YET", "SO", "BECAUSE", "ALTHOUGH", "WHILE", "UNLESS", "UNTIL", "SINCE"]},
            {"subgroup": "Relative pronouns", "words": ["WHO", "WHOM", "WHOSE", "WHICH", "THAT", "WHERE", "WHEN", "WHY", "WHAT", "WHATEVER", "WHOEVER", "WHICHEVER"]},
            {"subgroup": "Intensifiers", "words": ["VERY", "EXTREMELY", "REALLY", "QUITE", "RATHER", "FAIRLY", "PRETTY", "SOMEWHAT", "SLIGHTLY", "BARELY", "HARDLY"]},
            {"subgroup": "Quantifiers", "words": ["ALL", "SOME", "MANY", "FEW", "SEVERAL", "MOST", "ANY", "NO", "EVERY", "EACH", "EITHER", "NEITHER", "BOTH"]},
            {"subgroup": "Reflexive pronouns", "words": ["MYSELF", "YOURSELF", "HIMSELF", "HERSELF", "ITSELF", "OURSELVES", "YOURSELVES", "THEMSELVES", "ONESELF"]},
            {"subgroup": "Demonstratives", "words": ["THIS", "THAT", "THESE", "THOSE", "HERE", "THERE", "THEN", "NOW", "THUS", "HENCE", "THENCE", "SUCH"]}
        ]
    }
}

class LinguisticPuzzleGenerator:
    """
    Generator for creating complex linguistic reasoning puzzles
    """
    
    def __init__(self, patterns: Dict = None):
        """Initialize with linguistic patterns"""
        self.patterns = patterns or LINGUISTIC_PATTERNS
        self.used_patterns = []
        self.puzzle_history = []
    
    def get_pattern_info(self, pattern_name: str, index: int = None) -> Tuple[str, Optional[str]]:
        """Get pattern description and optional subgroup information"""
        pattern = self.patterns[pattern_name]
        
        if pattern.get("has_subgroups", False):
            if index is not None:
                example = pattern["examples"][index % len(pattern["examples"])]
                subgroup = example.get("subgroup", pattern_name)
                return pattern["description"], subgroup
        
        return pattern["description"], None
    
    def get_words_from_pattern(self, pattern_name: str, index: int = None) -> Tuple[List[str], str]:
        """Get words from a pattern, handling subgroups properly"""
        pattern = self.patterns[pattern_name]
        
        if pattern.get("has_subgroups", False):
            # For patterns with subgroups, pick a specific subgroup
            if index is None:
                index = random.randint(0, len(pattern["examples"]) - 1)
            example = pattern["examples"][index % len(pattern["examples"])]
            words = example["words"]
            subgroup = example.get("subgroup", pattern_name)
            return words, subgroup
        else:
            # For patterns without subgroups, pick from examples
            example = random.choice(pattern["examples"])
            if isinstance(example, dict):
                words = example["words"]
            else:
                words = example
            return words, pattern_name
    
    def generate_4_1_complex(self) -> Dict:
        """Generate 4:1 with one sophisticated pattern and one outlier"""
        # Pick two different complex patterns
        patterns = random.sample(list(self.patterns.keys()), 2)
        main_pattern = patterns[0]
        outlier_pattern = patterns[1]
        
        # Get words from main pattern with subgroup info
        main_words, main_subgroup = self.get_words_from_pattern(main_pattern)
        if len(main_words) < 4:
            return None
        main_words = random.sample(main_words, min(4, len(main_words)))
        
        # Get one word from outlier pattern
        outlier_words, outlier_subgroup = self.get_words_from_pattern(outlier_pattern)
        outlier = random.choice(outlier_words)
        
        all_words = main_words + [outlier]
        random.shuffle(all_words)
        
        target_scores = {word: (1 if word == outlier else 0) for word in all_words}
        
        return {
            "input": f"Pick the odd word out: {', '.join(all_words)}",
            "target_scores": target_scores,
            "pattern": "4:1",
            "explanation": f"Main: {main_subgroup}, Outlier: {outlier_subgroup}"
        }
    
    def generate_5_2_complex(self) -> Dict:
        """Generate 5:2 with sophisticated patterns"""
        patterns = random.sample(list(self.patterns.keys()), 2)
        main_pattern = patterns[0]
        minor_pattern = patterns[1]
        
        # Get words with subgroup info
        main_words, main_subgroup = self.get_words_from_pattern(main_pattern)
        minor_words, minor_subgroup = self.get_words_from_pattern(minor_pattern)
        
        # Sample safely
        if len(main_words) < 5 or len(minor_words) < 2:
            return None
        
        main_words = random.sample(main_words, 5)
        minor_words = random.sample(minor_words, 2)
        
        all_words = main_words + minor_words
        random.shuffle(all_words)
        
        target_scores = {word: (1 if word in minor_words else 0) for word in all_words}
        
        return {
            "input": f"Pick the odd words out: {', '.join(all_words)}",
            "target_scores": target_scores,
            "pattern": "5:2",
            "explanation": f"Main: {main_subgroup}, Minor: {minor_subgroup}"
        }
    
    def generate_7_3_complex(self) -> Dict:
        """Generate 7:3 with sophisticated patterns"""
        patterns = random.sample(list(self.patterns.keys()), 2)
        main_pattern = patterns[0]
        minor_pattern = patterns[1]
        
        # Get words with subgroup info
        main_words, main_subgroup = self.get_words_from_pattern(main_pattern)
        minor_words, minor_subgroup = self.get_words_from_pattern(minor_pattern)
        
        if len(main_words) < 7 or len(minor_words) < 3:
            return None
        
        main_words = random.sample(main_words, 7)
        minor_words = random.sample(minor_words, 3)
        
        all_words = main_words + minor_words
        random.shuffle(all_words)
        
        target_scores = {word: (1 if word in minor_words else 0) for word in all_words}
        
        return {
            "input": f"Pick the odd words out: {', '.join(all_words)}",
            "target_scores": target_scores,
            "pattern": "7:3",
            "explanation": f"Main: {main_subgroup}, Minor: {minor_subgroup}"
        }
    
    def generate_8_2_2_complex(self) -> Dict:
        """Generate 8:2:2 with three sophisticated patterns"""
        patterns = random.sample(list(self.patterns.keys()), 3)
        
        # Get words with subgroup info for each pattern
        main_words, main_subgroup = self.get_words_from_pattern(patterns[0])
        minor1_words, minor1_subgroup = self.get_words_from_pattern(patterns[1])
        minor2_words, minor2_subgroup = self.get_words_from_pattern(patterns[2])
        
        if len(main_words) < 8 or len(minor1_words) < 2 or len(minor2_words) < 2:
            return None
        
        main_words = random.sample(main_words, 8)
        minor1_words = random.sample(minor1_words, 2)
        minor2_words = random.sample(minor2_words, 2)
        
        all_words = main_words + minor1_words + minor2_words
        random.shuffle(all_words)
        
        target_scores = {}
        for word in all_words:
            if word in main_words:
                target_scores[word] = 0
            elif word in minor1_words:
                target_scores[word] = 1
            else:
                target_scores[word] = 2
        
        return {
            "input": f"There are 3 word groups, identify the word groups and their themes: {', '.join(all_words)}",
            "target_scores": target_scores,
            "pattern": "8:2:2",
            "explanation": f"Group 1 (main): {main_subgroup}, Group 2 (minor): {minor1_subgroup}, Group 3 (minor): {minor2_subgroup}"
        }
    
    def generate_10_3_3_complex(self) -> Dict:
        """Generate 10:3:3 with three sophisticated patterns"""
        patterns = random.sample(list(self.patterns.keys()), 3)
        
        # Get words with subgroup info for each pattern
        main_words, main_subgroup = self.get_words_from_pattern(patterns[0])
        minor1_words, minor1_subgroup = self.get_words_from_pattern(patterns[1])
        minor2_words, minor2_subgroup = self.get_words_from_pattern(patterns[2])
        
        if len(main_words) < 10 or len(minor1_words) < 3 or len(minor2_words) < 3:
            return None
        
        main_words = random.sample(main_words, 10)
        minor1_words = random.sample(minor1_words, 3)
        minor2_words = random.sample(minor2_words, 3)
        
        all_words = main_words + minor1_words + minor2_words
        random.shuffle(all_words)
        
        target_scores = {}
        for word in all_words:
            if word in main_words:
                target_scores[word] = 0
            elif word in minor1_words:
                target_scores[word] = 1
            else:
                target_scores[word] = 2
        
        return {
            "input": f"There are 3 word groups, identify the word groups and their themes: {', '.join(all_words)}",
            "target_scores": target_scores,
            "pattern": "10:3:3",
            "explanation": f"Group 1 (main): {main_subgroup}, Group 2 (minor): {minor1_subgroup}, Group 3 (minor): {minor2_subgroup}"
        }
    
    def generate_complex_examples(self, num_per_pattern: Dict[str, int]) -> List[Dict]:
        """Generate complex odd-one-out examples using sophisticated patterns"""
        all_examples = []
        
        for pattern_type, count in num_per_pattern.items():
            print(f"\nGenerating {count} examples of pattern {pattern_type}...")
            
            pattern_examples = []
            attempts = 0
            max_attempts = count * 10  # Allow plenty of retries
            
            while len(pattern_examples) < count and attempts < max_attempts:
                attempts += 1
                
                if len(pattern_examples) % 20 == 0 and len(pattern_examples) > 0:
                    print(f"  Generated {len(pattern_examples)}/{count}")
                
                if pattern_type == "4:1":
                    example = self.generate_4_1_complex()
                elif pattern_type == "5:2":
                    example = self.generate_5_2_complex()
                elif pattern_type == "7:3":
                    example = self.generate_7_3_complex()
                elif pattern_type == "8:2:2":
                    example = self.generate_8_2_2_complex()
                elif pattern_type == "10:3:3":
                    example = self.generate_10_3_3_complex()
                else:
                    continue
                
                if example:
                    pattern_examples.append(example)
                    all_examples.append(example)
            
            if len(pattern_examples) < count:
                print(f"  Warning: Only generated {len(pattern_examples)}/{count} after {attempts} attempts")
            
            print(f"  Completed: {len(pattern_examples)} examples")
        
        return all_examples
    
    def generate_puzzle(self, 
                        difficulty: str = "hard",
                        num_groups: int = 4,
                        words_per_group: int = 4,
                        include_distractors: bool = True) -> Dict:
        """
        Generate a linguistic reasoning puzzle
        
        Args:
            difficulty: Puzzle difficulty level
            num_groups: Number of groups to find
            words_per_group: Words in each group
            include_distractors: Whether to add misleading words
        
        Returns:
            Dictionary containing puzzle data
        """
        # Select pattern types based on difficulty
        if difficulty == "easy":
            pattern_types = random.sample(list(self.patterns.keys()), min(2, len(self.patterns)))
        elif difficulty == "medium":
            pattern_types = random.sample(list(self.patterns.keys()), min(3, len(self.patterns)))
        else:  # hard
            pattern_types = random.sample(list(self.patterns.keys()), min(4, len(self.patterns)))
        
        groups = []
        all_words = []
        explanations = []
        
        for i in range(num_groups):
            pattern_type = pattern_types[i % len(pattern_types)]
            pattern_data = self.patterns[pattern_type]
            
            # Select a specific example
            example = random.choice(pattern_data["examples"])
            
            # Get words for this group
            if pattern_data["has_subgroups"]:
                group_words = random.sample(example["words"], min(words_per_group, len(example["words"])))
                explanation = f"{pattern_type}: {example['subgroup']}"
            else:
                group_words = random.sample(example["words"], min(words_per_group, len(example["words"])))
                explanation = f"{pattern_type}: {pattern_data['description']}"
            
            groups.append(group_words)
            all_words.extend(group_words)
            explanations.append(explanation)
        
        # Add distractors if requested
        distractors = []
        if include_distractors:
            num_distractors = words_per_group  # Same number as a group
            distractor_patterns = random.sample(list(self.patterns.keys()), 2)
            
            for pattern_type in distractor_patterns:
                pattern_data = self.patterns[pattern_type]
                example = random.choice(pattern_data["examples"])
                
                if pattern_data["has_subgroups"]:
                    available_words = example["words"]
                else:
                    available_words = example["words"]
                
                # Get words not already used
                unused_words = [w for w in available_words if w not in all_words]
                if unused_words:
                    distractor_words = random.sample(unused_words, min(num_distractors // 2, len(unused_words)))
                    distractors.extend(distractor_words)
                    all_words.extend(distractor_words)
        
        # Shuffle all words
        random.shuffle(all_words)
        
        puzzle = {
            "puzzle_id": f"LING_{len(self.puzzle_history) + 1:04d}",
            "difficulty": difficulty,
            "words": all_words,
            "groups": groups,
            "explanations": explanations,
            "distractors": distractors,
            "pattern_types": pattern_types,
            "metadata": {
                "num_groups": num_groups,
                "words_per_group": words_per_group,
                "total_words": len(all_words),
                "has_distractors": include_distractors,
                "timestamp": str(Path(__file__).stat().st_mtime)
            }
        }
        
        self.puzzle_history.append(puzzle)
        return puzzle
    
    def generate_dataset(self, 
                        num_puzzles: int = 100,
                        output_path: str = "linguistic_puzzles.json") -> None:
        """
        Generate a dataset of linguistic puzzles
        
        Args:
            num_puzzles: Number of puzzles to generate
            output_path: Path to save the dataset
        """
        dataset = []
        difficulties = ["easy", "medium", "hard"]
        
        for i in range(num_puzzles):
            difficulty = difficulties[i % 3]
            
            # Vary parameters based on difficulty
            if difficulty == "easy":
                num_groups = random.choice([3, 4])
                words_per_group = 4
                include_distractors = random.choice([False, False, True])
            elif difficulty == "medium":
                num_groups = 4
                words_per_group = random.choice([4, 5])
                include_distractors = random.choice([True, True, False])
            else:  # hard
                num_groups = random.choice([4, 5])
                words_per_group = random.choice([4, 5, 6])
                include_distractors = True
            
            puzzle = self.generate_puzzle(
                difficulty=difficulty,
                num_groups=num_groups,
                words_per_group=words_per_group,
                include_distractors=include_distractors
            )
            
            dataset.append(puzzle)
            
            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_puzzles} puzzles...")
        
        # Save dataset
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
        print(f"Total puzzles: {len(dataset)}")
        print(f"Total unique patterns used: {len(set(sum([p['pattern_types'] for p in dataset], [])))}")
    
    def get_pattern_statistics(self) -> Dict:
        """Get statistics about available patterns"""
        stats = {
            "total_pattern_types": len(self.patterns),
            "total_subgroups": 0,
            "total_unique_words": set(),
            "pattern_details": {}
        }
        
        for pattern_type, pattern_data in self.patterns.items():
            subgroup_count = len(pattern_data["examples"])
            word_count = sum(len(ex.get("words", [])) for ex in pattern_data["examples"])
            
            stats["total_subgroups"] += subgroup_count
            stats["pattern_details"][pattern_type] = {
                "description": pattern_data["description"],
                "has_subgroups": pattern_data["has_subgroups"],
                "num_examples": subgroup_count,
                "total_words": word_count
            }
            
            for example in pattern_data["examples"]:
                stats["total_unique_words"].update(example.get("words", []))
        
        stats["total_unique_words"] = len(stats["total_unique_words"])
        return stats
    
    def main(self):
        """Main function to generate BigBench OOO dataset"""
        print("="*60)
        print("COMPREHENSIVE LINGUISTIC REASONING DATASET GENERATOR")
        print("NYT Connections-Style Patterns with Subgroup Tracking")
        print("="*60)
        
        Path('data/output').mkdir(parents=True, exist_ok=True)
        
        # Generate distribution
        distribution = {
            "4:1": 100,
            "5:2": 150,
            "7:3": 150,
            "8:2:2": 200,
            "10:3:3": 200
        }
        
        print("\nTarget distribution:")
        for pattern, count in distribution.items():
            print(f"  {pattern}: {count} examples")
        print(f"  Total: {sum(distribution.values())} examples")
        
        print(f"\nUsing {len(self.patterns)} complex pattern types with subgroup tracking")
        print("Pattern types:", ", ".join(list(self.patterns.keys())[:5]) + "...")
        
        # Generate examples
        all_examples = self.generate_complex_examples(distribution)
        
        # Filter out None values
        all_examples = [e for e in all_examples if e is not None]
        
        print(f"\nSuccessfully generated: {len(all_examples)} examples")
        
        # Save dataset
        dataset = {
            "task": "linguistic_reasoning_comprehensive",
            "description": "Comprehensive linguistic reasoning with NYT Connections-style patterns and proper subgroup tracking",
            "total_examples": len(all_examples),
            "pattern_types": list(self.patterns.keys()),
            "examples": all_examples
        }
        
        with open('data/output/preconn_raw.json', 'w') as f:
            json.dump(dataset, f, indent=2)
        
        # Create JSONL for training
        with open('data/output/preconn_raw.jsonl', 'w') as f:
            for ex in all_examples:
                # Format answer based on pattern
                if ex["pattern"] in ["4:1", "5:2", "7:3"]:
                    odd_words = [k for k, v in ex['target_scores'].items() if v == 1]
                    answer = f"The odd word(s) out: {', '.join(odd_words)}"
                else:
                    groups = {}
                    for word, group_id in ex['target_scores'].items():
                        if group_id not in groups:
                            groups[group_id] = []
                        groups[group_id].append(word)
                    
                    answer_parts = []
                    if 0 in groups:
                        answer_parts.append(f"Main group: {', '.join(groups[0])}")
                    if 1 in groups:
                        answer_parts.append(f"Minor group 1: {', '.join(groups[1])}")
                    if 2 in groups:
                        answer_parts.append(f"Minor group 2: {', '.join(groups[2])}")
                    answer = "\n".join(answer_parts)
                
                f.write(json.dumps({
                    "messages": [
                        {"role": "user", "content": ex["input"]},
                        {"role": "assistant", "content": answer}
                    ],
                    "metadata": {
                        "pattern": ex["pattern"],
                        "explanation": ex.get("explanation", "")
                    }
                }) + '\n')
        
        print("\nFiles created:")
        print("  - data/output/linguistic_reasoning_comprehensive.json")
        print("  - data/output/linguistic_reasoning_comprehensive.jsonl")
        
        # Print pattern statistics
        print("\nPattern usage statistics (with subgroups):")
        pattern_counts = {}
        for ex in all_examples:
            exp = ex.get("explanation", "")
            # Count actual subgroup usage
            parts = exp.split(", ")
            for part in parts:
                if ":" in part:
                    _, subgroup = part.split(": ", 1)
                    pattern_counts[subgroup] = pattern_counts.get(subgroup, 0) + 1
        
        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {pattern}: {count} uses")
        
        # Print some examples
        print("\nSample complex examples with subgroup info:")
        for i in range(min(5, len(all_examples))):
            ex = all_examples[i]
            print(f"\nExample {i+1} ({ex['pattern']}):")
            print(f"  Pattern details: {ex.get('explanation', 'N/A')}")
            print(f"  Input: {ex['input'][:100]}...")


# Example usage
if __name__ == "__main__":
    generator = LinguisticPuzzleGenerator()
    
    # Run the main dataset generation
    generator.main()
    
    # Also demonstrate individual puzzle generation
    print("\n" + "="*60)
    print("INDIVIDUAL PUZZLE GENERATION DEMO")
    print("="*60)
    
    # Generate a single puzzle
    puzzle = generator.generate_puzzle(difficulty="hard")
    print("\nSample Puzzle:")
    print(f"Words: {puzzle['words']}")
    print(f"Groups to find: {len(puzzle['groups'])}")
    print(f"Pattern types used: {puzzle['pattern_types']}")
    print()
    
    # Get statistics
    stats = generator.get_pattern_statistics()
    print("Dataset Statistics:")
    print(f"Total pattern types: {stats['total_pattern_types']}")
    print(f"Total subgroups: {stats['total_subgroups']}")
    print(f"Total unique words: {stats['total_unique_words']}")