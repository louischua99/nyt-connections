"""
categorical linguistic reasoning dataset generator
creates puzzles based on analyzed nyt connections patterns
uses real pattern categories from gpt-5 taxonomy analysis
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# nyt connections pattern categories from analysis
CATEGORICAL_PATTERNS = {
    "SEMANTIC_TAXONOMY": {
        "description": "items are types of x, parts of y, members/components of conventional category",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "pasta shapes", "words": ["PENNE", "RIGATONI", "FUSILLI", "FARFALLE", "ORZO", "ROTINI", "ZITI", "LINGUINE", "FETTUCCINE", "BOWTIE", "SHELLS", "MACARONI"]},
            {"subgroup": "cloud types", "words": ["CUMULUS", "CIRRUS", "STRATUS", "NIMBUS", "ALTO", "CUMULONIMBUS", "CIRROSTRATUS", "ALTOSTRATUS", "NIMBOSTRATUS", "CIRROCUMULUS", "STRATOCUMULUS", "ALTOCUMULUS"]},
            {"subgroup": "knots", "words": ["BOWLINE", "CLOVE", "REEF", "SHEET", "SQUARE", "GRANNY", "FIGURE", "SLIP", "STOPPER", "TIMBER", "ROLLING", "ANCHOR"]},
            {"subgroup": "coffee drinks", "words": ["ESPRESSO", "AMERICANO", "CAPPUCCINO", "LATTE", "MACCHIATO", "CORTADO", "DOPPIO", "LUNGO", "RISTRETTO", "AFFOGATO", "MOCHA", "BREVE"]},
            {"subgroup": "poker hands", "words": ["FLUSH", "STRAIGHT", "PAIR", "TRIPS", "QUADS", "BOAT", "WHEEL", "BROADWAY", "NUTS", "ROYAL", "LOWBALL", "RAZZ"]},
            {"subgroup": "wind instruments", "words": ["FLUTE", "OBOE", "CLARINET", "BASSOON", "SAXOPHONE", "TRUMPET", "TROMBONE", "TUBA", "HORN", "PICCOLO", "RECORDER", "HARMONICA"]},
            {"subgroup": "gem cuts", "words": ["ROUND", "PRINCESS", "EMERALD", "CUSHION", "OVAL", "MARQUISE", "PEAR", "RADIANT", "ASSCHER", "HEART", "BAGUETTE", "TRILLION"]},
            {"subgroup": "bread types", "words": ["SOURDOUGH", "RYE", "PUMPERNICKEL", "CIABATTA", "FOCACCIA", "BAGUETTE", "BRIOCHE", "CHALLAH", "NAAN", "PITA", "TORTILLA", "LAVASH"]},
            {"subgroup": "garden tools", "words": ["RAKE", "HOE", "SPADE", "TROWEL", "CULTIVATOR", "PRUNER", "SHEARS", "FORK", "EDGER", "WEEDER", "DIBBER", "MATTOCK"]},
            {"subgroup": "sailing terms", "words": ["PORT", "STARBOARD", "BOW", "STERN", "MAST", "BOOM", "KEEL", "RUDDER", "TILLER", "SHEET", "HALYARD", "CLEAT"]},
            {"subgroup": "chess terms", "words": ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING", "CASTLE", "CHECK", "MATE", "STALEMATE", "ENDGAME", "OPENING"]},
            {"subgroup": "fabric weaves", "words": ["PLAIN", "TWILL", "SATIN", "BASKET", "HERRINGBONE", "HOUNDSTOOTH", "CHEVRON", "OXFORD", "POPLIN", "CHAMBRAY", "DENIM", "CANVAS"]},
            {"subgroup": "cooking methods", "words": ["SAUTE", "BRAISE", "ROAST", "GRILL", "BROIL", "STEAM", "POACH", "SIMMER", "BOIL", "FRY", "BAKE", "SMOKE"]},
            {"subgroup": "yoga poses", "words": ["DOWNWARD", "WARRIOR", "TREE", "CHILD", "COBRA", "PLANK", "BRIDGE", "TRIANGLE", "PIGEON", "LOTUS", "CORPSE", "MOUNTAIN"]},
            {"subgroup": "percussion instruments", "words": ["SNARE", "BASS", "TIMPANI", "CYMBAL", "TRIANGLE", "TAMBOURINE", "MARACAS", "BONGO", "CONGA", "DJEMBE", "TABLA", "CAJON"]},
            {"subgroup": "wine varieties", "words": ["MERLOT", "CABERNET", "PINOT", "CHARDONNAY", "RIESLING", "SYRAH", "MALBEC", "ZINFANDEL", "MOSCATO", "PROSECCO", "CHIANTI", "BORDEAUX"]},
            {"subgroup": "architectural styles", "words": ["GOTHIC", "BAROQUE", "ROCOCO", "NEOCLASSICAL", "VICTORIAN", "MODERN", "BRUTALIST", "COLONIAL", "TUDOR", "CRAFTSMAN", "RANCH", "CAPE"]},
            {"subgroup": "martial arts", "words": ["KARATE", "JUDO", "AIKIDO", "TAEKWONDO", "JUJITSU", "HAPKIDO", "BOXING", "WRESTLING", "CAPOEIRA", "KENDO", "FENCING", "SAVATE"]},
            {"subgroup": "cloud services", "words": ["STORAGE", "COMPUTE", "DATABASE", "NETWORK", "SECURITY", "ANALYTICS", "MACHINE", "CONTAINER", "SERVERLESS", "BACKUP", "MONITORING", "IDENTITY"]},
            {"subgroup": "photography terms", "words": ["APERTURE", "SHUTTER", "ISO", "EXPOSURE", "BOKEH", "DEPTH", "FOCAL", "COMPOSITION", "FRAMING", "CONTRAST", "SATURATION", "VIGNETTE"]}
        ]
    },

    "SEMANTIC_SYNONYMY": {
        "description": "words with closely related meanings, idiomatic paraphrases sharing sense",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "intelligence", "words": ["SMART", "CLEVER", "BRILLIANT", "BRIGHT", "SHARP", "QUICK", "ASTUTE", "KEEN", "ACUTE", "PERCEPTIVE", "SHREWD", "WISE"]},
            {"subgroup": "small", "words": ["TINY", "LITTLE", "MINUTE", "MINIATURE", "PETITE", "DIMINUTIVE", "MICROSCOPIC", "MINUSCULE", "SLIGHT", "MODEST", "COMPACT", "MINI"]},
            {"subgroup": "large", "words": ["HUGE", "GIANT", "MASSIVE", "ENORMOUS", "IMMENSE", "VAST", "COLOSSAL", "MAMMOTH", "TREMENDOUS", "GIGANTIC", "MONUMENTAL", "TITANIC"]},
            {"subgroup": "fast", "words": ["QUICK", "RAPID", "SWIFT", "SPEEDY", "HASTY", "BRISK", "FLEET", "NIMBLE", "PROMPT", "EXPEDITIOUS", "ACCELERATED", "EXPRESS"]},
            {"subgroup": "difficult", "words": ["HARD", "TOUGH", "CHALLENGING", "ARDUOUS", "DEMANDING", "RIGOROUS", "STRENUOUS", "TAXING", "LABORIOUS", "GRUELING", "FORMIDABLE", "DAUNTING"]},
            {"subgroup": "beautiful", "words": ["PRETTY", "LOVELY", "GORGEOUS", "STUNNING", "ATTRACTIVE", "HANDSOME", "ELEGANT", "EXQUISITE", "RADIANT", "MAGNIFICENT", "SPLENDID", "STRIKING"]},
            {"subgroup": "angry", "words": ["MAD", "FURIOUS", "IRATE", "LIVID", "ENRAGED", "INCENSED", "OUTRAGED", "INFURIATED", "WRATHFUL", "INDIGNANT", "CROSS", "UPSET"]},
            {"subgroup": "happy", "words": ["JOYFUL", "CHEERFUL", "GLAD", "PLEASED", "DELIGHTED", "ELATED", "ECSTATIC", "JUBILANT", "THRILLED", "CONTENT", "MERRY", "UPBEAT"]},
            {"subgroup": "sad", "words": ["UNHAPPY", "MISERABLE", "SORROWFUL", "MELANCHOLY", "GLOOMY", "DEJECTED", "DESPONDENT", "DOWNCAST", "FORLORN", "WOEFUL", "GLUM", "BLUE"]},
            {"subgroup": "strange", "words": ["ODD", "WEIRD", "BIZARRE", "PECULIAR", "CURIOUS", "UNUSUAL", "ECCENTRIC", "QUIRKY", "ABNORMAL", "UNCANNY", "EERIE", "OUTLANDISH"]},
            {"subgroup": "important", "words": ["CRUCIAL", "VITAL", "ESSENTIAL", "CRITICAL", "SIGNIFICANT", "MAJOR", "PARAMOUNT", "FUNDAMENTAL", "PIVOTAL", "INTEGRAL", "CENTRAL", "PRIMARY"]},
            {"subgroup": "wealthy", "words": ["RICH", "AFFLUENT", "PROSPEROUS", "OPULENT", "LOADED", "FLUSH", "MONEYED", "WELL-OFF", "COMFORTABLE", "PRIVILEGED", "ELITE", "UPSCALE"]},
            {"subgroup": "dangerous", "words": ["HAZARDOUS", "PERILOUS", "RISKY", "TREACHEROUS", "PRECARIOUS", "UNSAFE", "THREATENING", "MENACING", "DIRE", "GRAVE", "SERIOUS", "CRITICAL"]},
            {"subgroup": "funny", "words": ["HILARIOUS", "AMUSING", "COMICAL", "HUMOROUS", "WITTY", "ENTERTAINING", "HYSTERICAL", "RIDICULOUS", "ABSURD", "LAUGHABLE", "DROLL", "WHIMSICAL"]},
            {"subgroup": "tired", "words": ["EXHAUSTED", "WEARY", "FATIGUED", "DRAINED", "SPENT", "WORN", "BEAT", "BUSHED", "POOPED", "WIPED", "DEPLETED", "DROWSY"]},
            {"subgroup": "fake", "words": ["FALSE", "PHONY", "BOGUS", "COUNTERFEIT", "FRAUDULENT", "SHAM", "SPURIOUS", "MOCK", "ARTIFICIAL", "SYNTHETIC", "IMITATION", "FORGED"]}
        ]
    },

    "SEMANTIC_ASSOCIATION": {
        "description": "items linked by shared attribute, function, scenario without being synonyms",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "camping equipment", "words": ["TENT", "BEDROLL", "CANTEEN", "LANTERN", "STOVE", "COOLER", "BACKPACK", "COMPASS", "FLASHLIGHT", "MATCHES", "ROPE", "TARP"]},
            {"subgroup": "birthday party", "words": ["CAKE", "CANDLES", "PRESENTS", "BALLOONS", "STREAMERS", "GUESTS", "GAMES", "MUSIC", "FOOD", "DECORATIONS", "INVITATIONS", "FAVORS"]},
            {"subgroup": "detective story", "words": ["CLUE", "SUSPECT", "MOTIVE", "ALIBI", "EVIDENCE", "WITNESS", "CRIME", "SCENE", "FINGERPRINT", "INTERROGATION", "CONFESSION", "ARREST"]},
            {"subgroup": "hospital items", "words": ["STETHOSCOPE", "SYRINGE", "BANDAGE", "GURNEY", "CHART", "SCALPEL", "GLOVES", "MASK", "GOWN", "MONITOR", "WHEELCHAIR", "CRUTCHES"]},
            {"subgroup": "beach day", "words": ["SAND", "WAVES", "UMBRELLA", "TOWEL", "SUNSCREEN", "SWIMSUIT", "BUCKET", "SHOVEL", "SEASHELLS", "SANDCASTLE", "LIFEGUARD", "BOARDWALK"]},
            {"subgroup": "office supplies", "words": ["STAPLER", "PAPERCLIP", "PRINTER", "COMPUTER", "DESK", "CHAIR", "FOLDER", "BINDER", "TELEPHONE", "CALCULATOR", "NOTEBOOK", "WHITEBOARD"]},
            {"subgroup": "kitchen appliances", "words": ["REFRIGERATOR", "STOVE", "MICROWAVE", "DISHWASHER", "BLENDER", "TOASTER", "MIXER", "PROCESSOR", "KETTLE", "COFFEEMAKER", "JUICER", "GRIDDLE"]},
            {"subgroup": "weather phenomena", "words": ["THUNDER", "LIGHTNING", "RAINBOW", "TORNADO", "HURRICANE", "BLIZZARD", "DROUGHT", "FLOOD", "HEATWAVE", "FOG", "FROST", "HAIL"]},
            {"subgroup": "school supplies", "words": ["PENCIL", "ERASER", "RULER", "BACKPACK", "TEXTBOOK", "CALCULATOR", "HIGHLIGHTER", "BINDER", "GLUE", "SCISSORS", "CRAYON", "MARKER"]},
            {"subgroup": "gym equipment", "words": ["DUMBBELL", "BARBELL", "TREADMILL", "ELLIPTICAL", "BENCH", "MAT", "KETTLEBELL", "ROPE", "BALL", "WEIGHTS", "BANDS", "ROLLER"]},
            {"subgroup": "farm animals", "words": ["COW", "PIG", "CHICKEN", "SHEEP", "GOAT", "HORSE", "DUCK", "TURKEY", "DONKEY", "GOOSE", "RABBIT", "LLAMA"]},
            {"subgroup": "space objects", "words": ["PLANET", "STAR", "MOON", "ASTEROID", "COMET", "GALAXY", "NEBULA", "METEOR", "SATELLITE", "BLACKHOLE", "QUASAR", "PULSAR"]},
            {"subgroup": "circus acts", "words": ["CLOWN", "JUGGLER", "ACROBAT", "TRAPEZE", "TIGHTROPE", "RINGMASTER", "ANIMAL", "TRAINER", "MAGICIAN", "STRONGMAN", "CONTORTIONIST", "UNICYCLIST"]},
            {"subgroup": "construction site", "words": ["CRANE", "BULLDOZER", "EXCAVATOR", "CONCRETE", "STEEL", "BEAM", "SCAFFOLD", "HARDHAT", "HELMET", "BARRIER", "BLUEPRINT", "FOUNDATION"]},
            {"subgroup": "orchestra sections", "words": ["STRINGS", "BRASS", "WOODWIND", "PERCUSSION", "CONDUCTOR", "BATON", "SCORE", "STAND", "PODIUM", "REHEARSAL", "CONCERT", "SYMPHONY"]}
        ]
    },

    "NAMED_ENTITIES": {
        "description": "proper names: people, characters, places, brands, titles, teams",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "shakespeare plays", "words": ["HAMLET", "MACBETH", "OTHELLO", "LEAR", "CORIOLANUS", "CYMBELINE", "TEMPEST", "PERICLES", "TIMON", "TITUS", "JULIUS", "RICHARD"]},
            {"subgroup": "olympic cities", "words": ["PARIS", "LONDON", "TOKYO", "BEIJING", "ATHENS", "SYDNEY", "ATLANTA", "BARCELONA", "SEOUL", "MOSCOW", "MONTREAL", "MUNICH"]},
            {"subgroup": "tech companies", "words": ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON", "META", "TESLA", "NETFLIX", "INTEL", "NVIDIA", "ORACLE", "ADOBE", "SPOTIFY"]},
            {"subgroup": "greek gods", "words": ["ZEUS", "HERA", "APOLLO", "ATHENA", "POSEIDON", "ARTEMIS", "HERMES", "APHRODITE", "ARES", "HADES", "DEMETER", "DIONYSUS"]},
            {"subgroup": "car brands", "words": ["TOYOTA", "FORD", "HONDA", "TESLA", "BMW", "MERCEDES", "AUDI", "VOLKSWAGEN", "NISSAN", "HYUNDAI", "MAZDA", "SUBARU"]},
            {"subgroup": "superheroes", "words": ["SUPERMAN", "BATMAN", "SPIDERMAN", "IRONMAN", "HULK", "THOR", "HAWKEYE", "VISION", "FLASH", "AQUAMAN", "WOLVERINE", "DEADPOOL"]},
            {"subgroup": "disney princesses", "words": ["ELSA", "CINDERELLA", "AURORA", "ARIEL", "BELLE", "JASMINE", "POCAHONTAS", "MULAN", "TIANA", "RAPUNZEL", "MERIDA", "MOANA"]},
            {"subgroup": "european capitals", "words": ["PARIS", "LONDON", "BERLIN", "ROME", "MADRID", "VIENNA", "PRAGUE", "AMSTERDAM", "BRUSSELS", "LISBON", "STOCKHOLM", "COPENHAGEN"]},
            {"subgroup": "famous painters", "words": ["PICASSO", "MONET", "VINCI", "GOGH", "REMBRANDT", "MICHELANGELO", "POLLOCK", "WARHOL", "DALI", "KAHLO", "MATISSE", "CEZANNE"]},
            {"subgroup": "elite universities", "words": ["HARVARD", "YALE", "PRINCETON", "COLUMBIA", "BROWN", "CORNELL", "DARTMOUTH", "PENN", "STANFORD", "MIT", "DUKE", "CHICAGO"]},
            {"subgroup": "beatles songs", "words": ["YESTERDAY", "HELP", "SOMETHING", "ELEANOR", "LUCY", "ABBEY", "YELLOW", "PENNY", "REVOLUTION", "BLACKBIRD", "MICHELLE", "TAXMAN"]},
            {"subgroup": "bond villains", "words": ["GOLDFINGER", "BLOFELD", "SCARAMANGA", "DRAX", "ZORIN", "SANCHEZ", "TREVELYAN", "CARVER", "RENARD", "GRAVES", "GREENE", "SILVA"]},
            {"subgroup": "fashion designers", "words": ["CHANEL", "DIOR", "VERSACE", "GUCCI", "PRADA", "ARMANI", "VALENTINO", "HERMES", "VUITTON", "BALENCIAGA", "BURBERRY", "GIVENCHY"]},
            {"subgroup": "nba teams", "words": ["LAKERS", "CELTICS", "BULLS", "WARRIORS", "HEAT", "SPURS", "CAVALIERS", "ROCKETS", "NETS", "CLIPPERS", "BUCKS", "SUNS"]}
        ]
    },

    "COLLOCATIONAL_IDIOMATIC": {
        "description": "items fill common slot in fixed phrases, compounds, idioms",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "___cake", "words": ["CUP", "PAN", "POUND", "CHEESE", "CARROT", "COFFEE", "BIRTH", "WEDDING", "FRUIT", "SPONGE", "LAYER", "BUNDT"]},
            {"subgroup": "___storm", "words": ["THUNDER", "RAIN", "SNOW", "ICE", "DUST", "SAND", "WIND", "HAIL", "BRAIN", "FIRE", "PERFECT", "TROPICAL"]},
            {"subgroup": "___house", "words": ["GREEN", "WHITE", "LIGHT", "COURT", "WARE", "PLAY", "DOG", "TREE", "GUEST", "BOAT", "FARM", "TOWN"]},
            {"subgroup": "rock___", "words": ["STAR", "BOTTOM", "SOLID", "HARD", "STEADY", "GARDEN", "FACE", "CLIMBING", "SALT", "MUSIC", "BAND", "CONCERT"]},
            {"subgroup": "___line", "words": ["DEAD", "FINISH", "STARTING", "BOTTOM", "FRONT", "PICKET", "ASSEMBLY", "CLOTHES", "FISHING", "PHONE", "PUNCH", "STORY"]},
            {"subgroup": "___board", "words": ["KEY", "SURF", "SKATE", "SNOW", "DASH", "CLIP", "BILL", "CARD", "FLOOR", "SWITCH", "MOTHER", "SCORE"]},
            {"subgroup": "___room", "words": ["BED", "BATH", "CLASS", "LIVING", "DINING", "BOARD", "CHAT", "SHOW", "MUSH", "DARK", "EMERGENCY", "WAITING"]},
            {"subgroup": "fire___", "words": ["PLACE", "WORKS", "FIGHTER", "TRUCK", "ALARM", "ESCAPE", "PROOF", "WALL", "STATION", "HYDRANT", "EXTINGUISHER", "DRILL"]},
            {"subgroup": "___light", "words": ["SUN", "MOON", "STAR", "CANDLE", "FLASH", "SPOT", "HEAD", "TAIL", "STOP", "GREEN", "HIGH", "LOW"]},
            {"subgroup": "___work", "words": ["HOME", "NET", "FRAME", "FIRE", "TEAM", "HARD", "BUSY", "PAPER", "FIELD", "GROUND", "PATCH", "ART"]},
            {"subgroup": "___ball", "words": ["FOOT", "BASE", "BASKET", "VOLLEY", "SOFT", "HAND", "EYE", "SNOW", "MEAT", "GOLF", "TENNIS", "BOWLING"]},
            {"subgroup": "___bird", "words": ["BLACK", "BLUE", "HUMMING", "MOCKING", "SONG", "LOVE", "THUNDER", "JAIL", "EARLY", "SNOW", "SEA", "LADY"]},
            {"subgroup": "___fish", "words": ["GOLD", "CAT", "SWORD", "STAR", "JELLY", "SHELL", "BLOW", "ANGEL", "CLOWN", "SILVER", "FLYING", "FIGHTING"]},
            {"subgroup": "___land", "words": ["HIGH", "LOW", "MAIN", "HOME", "WASTE", "WET", "DRY", "FARM", "GRASS", "WOOD", "WONDER", "DREAM"]},
            {"subgroup": "___ship", "words": ["FRIEND", "RELATION", "PARTNER", "SCHOLAR", "CHAMPION", "APPRENTICE", "LEADER", "OWNER", "MEMBER", "CITIZEN", "FELLOW", "SPONSOR"]}
        ]
    },

    "LEXICAL_MORPHOLOGY": {
        "description": "shared word formation: affixation, compounding, clippings, blends",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "anti___ prefix", "words": ["VIRUS", "BACTERIAL", "INFLAMMATORY", "OXIDANT", "DEPRESSANT", "FREEZE", "THEFT", "AGING", "GRAVITY", "SOCIAL", "CLIMACTIC", "HERO"]},
            {"subgroup": "___ology suffix", "words": ["BIO", "PSYCHO", "SOCIO", "ANTHROPO", "ARCHAEO", "TECHNO", "METHODO", "IDEO", "CHRONO", "ETYMO", "COSMO", "THEO"]},
            {"subgroup": "super___ prefix", "words": ["MARKET", "HERO", "NATURAL", "SONIC", "SIZED", "STAR", "COMPUTER", "POWER", "CHARGED", "HIGHWAY", "MODEL", "STORE"]},
            {"subgroup": "___ism suffix", "words": ["CAPITAL", "SOCIAL", "COMMUN", "REAL", "IDEAL", "NATIONAL", "LIBERAL", "CONSERVAT", "MODERN", "TRADITIONAL", "COLONIAL", "IMPERIAL"]},
            {"subgroup": "inter___ prefix", "words": ["NATIONAL", "NET", "VIEW", "FACE", "ACT", "VENE", "CEPT", "CHANGE", "SECTION", "STATE", "PERSONAL", "STELLAR"]},
            {"subgroup": "___tion suffix", "words": ["EDUCA", "INFORMA", "COMMUNICA", "TRANSPORTA", "ADMINISTRA", "CELEBRA", "CONCENTRA", "DEMONSTRA", "ILLUSTRA", "REGISTRA", "COOPERA", "INVESTIGA"]},
            {"subgroup": "multi___ prefix", "words": ["CULTURAL", "NATIONAL", "MEDIA", "TASKING", "PURPOSE", "PLAYER", "LEVEL", "COLORED", "LINGUAL", "VITAMIN", "MILLION", "PLATFORM"]},
            {"subgroup": "___ness suffix", "words": ["DARK", "BRIGHT", "SAD", "HAPPY", "WEAK", "STRONG", "THICK", "THIN", "ROUGH", "SMOOTH", "SOFT", "HARD"]},
            {"subgroup": "non___ prefix", "words": ["SENSE", "FICTION", "PROFIT", "STOP", "STICK", "TOXIC", "VERBAL", "LINEAR", "BINARY", "ESSENTIAL", "VIOLENT", "SMOKING"]},
            {"subgroup": "___able suffix", "words": ["COMFORT", "FASHION", "REASON", "NOTICE", "REMARK", "CONSIDER", "ACCEPT", "ENJOY", "PREFER", "ADMIRE", "DESIRE", "RESPECT"]},
            {"subgroup": "trans___ prefix", "words": ["PORT", "FORM", "PLANT", "FER", "MIT", "LATE", "PARENT", "ATLANTIC", "PACIFIC", "CONTINENTAL", "NATIONAL", "GENDER"]},
            {"subgroup": "___ful suffix", "words": ["CARE", "HOPE", "PEACE", "JOY", "THANK", "THOUGHT", "WONDER", "BEAUTY", "PAIN", "USE", "HELP", "POWER"]}
        ]
    },

    "LEXICAL_ORTHOGRAPHY": {
        "description": "visible letter patterns: anagrams, palindromes, substrings, letter edits",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "words containing 'oo'", "words": ["BOOK", "COOK", "LOOK", "TOOK", "MOON", "SOON", "NOON", "FOOD", "GOOD", "WOOD", "HOOD", "STOOD"]},
            {"subgroup": "palindromes", "words": ["KAYAK", "LEVEL", "RADAR", "ROTOR", "CIVIC", "MADAM", "REFER", "STATS", "NOON", "DEED", "TENET", "SAGAS"]},
            {"subgroup": "double consonants", "words": ["LETTER", "BUTTER", "COFFEE", "MIDDLE", "BUBBLE", "PUZZLE", "SCISSORS", "GRAMMAR", "SUMMER", "PEPPER", "PILLOW", "BOTTLE"]},
            {"subgroup": "silent b words", "words": ["LAMB", "THUMB", "COMB", "CLIMB", "BOMB", "TOMB", "PLUMBER", "CRUMB", "DEBT", "DOUBT", "SUBTLE", "LIMB"]},
            {"subgroup": "words ending in 'tion'", "words": ["NATION", "STATION", "ACTION", "FRACTION", "VACATION", "EDUCATION", "CREATION", "SOLUTION", "POLLUTION", "TRADITION", "CONDITION", "POSITION"]},
            {"subgroup": "words starting with 'qu'", "words": ["QUEEN", "QUICK", "QUIET", "QUITE", "QUOTE", "QUEST", "QUILT", "QUARTER", "QUALITY", "QUANTITY", "QUESTION", "QUARREL"]},
            {"subgroup": "words with 'ph' as 'f'", "words": ["PHONE", "PHOTO", "GRAPH", "PHRASE", "PHANTOM", "PHOENIX", "PHARMACY", "ALPHABET", "ELEPHANT", "TROPHY", "NEPHEW", "PROPHET"]},
            {"subgroup": "anagrams of 'stop'", "words": ["STOP", "TOPS", "POTS", "SPOT", "POST", "OPTS", "TOPS", "POTS", "SPOT", "POST", "OPTS", "STOP"]},
            {"subgroup": "words with all vowels", "words": ["FACETIOUS", "ABSTEMIOUS", "ARSENIOUS", "ABSTENTIOUS", "EDUCATION", "SEQUOIA", "EUPHORIA", "DIALOGUE", "EQUATION", "AUCTIONED", "CAUTIONED", "AUTOMOBILE"]},
            {"subgroup": "words ending in 'x'", "words": ["BOX", "FOX", "MIX", "FIX", "SIX", "WAX", "TAX", "RELAX", "INDEX", "COMPLEX", "REFLEX", "APEX"]},
            {"subgroup": "contractions without apostrophe", "words": ["CANT", "WONT", "DONT", "DOESNT", "DIDNT", "WOULDNT", "COULDNT", "SHOULDNT", "HASNT", "HAVENT", "ISNT", "ARENT"]},
            {"subgroup": "words with 'ght'", "words": ["LIGHT", "NIGHT", "RIGHT", "FIGHT", "MIGHT", "SIGHT", "BRIGHT", "FLIGHT", "HEIGHT", "WEIGHT", "TIGHT", "SLIGHT"]}
        ]
    },

    "PHONOLOGICAL_PATTERN": {
        "description": "sound relations: rhyme, homophones, silent letters, alliteration",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "rhymes with 'eat'", "words": ["BEAT", "HEAT", "MEAT", "SEAT", "TREAT", "WHEAT", "CHEAT", "FEAT", "NEAT", "SWEET", "FLEET", "SHEET"]},
            {"subgroup": "homophones of numbers", "words": ["ONE", "WON", "TWO", "TOO", "TO", "FOUR", "FOR", "FORE", "EIGHT", "ATE", "SUM", "SOME"]},
            {"subgroup": "silent k words", "words": ["KNIFE", "KNEE", "KNOW", "KNOT", "KNIGHT", "KNIT", "KNOCK", "KNUCKLE", "KNOWLEDGE", "KNEAD", "KNAPSACK", "KNACK"]},
            {"subgroup": "words with long 'a' sound", "words": ["MAKE", "TAKE", "CAKE", "LAKE", "BRAKE", "SNAKE", "STAKE", "WAKE", "FAKE", "BAKE", "SHAKE", "FLAKE"]},
            {"subgroup": "alliteration with 'b'", "words": ["BIG", "BAD", "BOLD", "BRAVE", "BRIGHT", "BEAUTIFUL", "BUSY", "BITTER", "BETTER", "BROKEN", "BRIEF", "BROAD"]},
            {"subgroup": "words ending in 'ough'", "words": ["THOUGH", "THROUGH", "TOUGH", "ROUGH", "ENOUGH", "COUGH", "BOUGH", "DOUGH", "PLOUGH", "TROUGH", "DROUGHT", "BOUGHT"]},
            {"subgroup": "silent l words", "words": ["WALK", "TALK", "CHALK", "STALK", "FOLK", "YOLK", "HALF", "CALF", "SALMON", "CALM", "PALM", "BALM"]},
            {"subgroup": "homophones of body parts", "words": ["HEEL", "HEAL", "SOLE", "SOUL", "HAIR", "HARE", "MUSCLE", "MUSSEL", "NAVAL", "NAVEL", "VAIN", "VEIN"]},
            {"subgroup": "rhymes with 'all'", "words": ["BALL", "CALL", "FALL", "HALL", "MALL", "TALL", "WALL", "SMALL", "STALL", "CRAWL", "BRAWL", "SPRAWL"]},
            {"subgroup": "words with 'ie' as long 'e'", "words": ["CHIEF", "BRIEF", "FIELD", "YIELD", "SHIELD", "PIECE", "NIECE", "BELIEVE", "ACHIEVE", "RECEIVE", "DECEIVE", "CONCEIVE"]},
            {"subgroup": "silent w words", "words": ["WRITE", "WRONG", "WRAP", "WRIST", "WRECK", "WRESTLE", "WRINKLE", "WREATH", "WRATH", "WRENCH", "WRIGGLE", "WRITTEN"]},
            {"subgroup": "words with soft 'g'", "words": ["GIANT", "GENTLE", "GENIUS", "GENERAL", "GENERATE", "GENUINE", "GESTURE", "GIRAFFE", "GINGER", "GYMNAST", "GEOLOGY", "GEOGRAPHY"]}
        ]
    },

    "GRAMMATICAL_SYNTACTIC": {
        "description": "grouping by part of speech, inflection, discourse function",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "irregular past tense", "words": ["WENT", "CAME", "SAW", "TOOK", "GAVE", "MADE", "SAID", "TOLD", "THOUGHT", "BROUGHT", "CAUGHT", "TAUGHT"]},
            {"subgroup": "modal verbs", "words": ["CAN", "COULD", "MAY", "MIGHT", "MUST", "SHALL", "SHOULD", "WILL", "WOULD", "OUGHT", "NEED", "DARE"]},
            {"subgroup": "prepositions of place", "words": ["IN", "ON", "AT", "UNDER", "OVER", "BEHIND", "BESIDE", "BETWEEN", "AMONG", "THROUGH", "ACROSS", "ALONG"]},
            {"subgroup": "conjunctions", "words": ["AND", "BUT", "OR", "NOR", "FOR", "YET", "SO", "ALTHOUGH", "BECAUSE", "SINCE", "WHILE", "UNLESS"]},
            {"subgroup": "reflexive pronouns", "words": ["MYSELF", "YOURSELF", "HIMSELF", "HERSELF", "ITSELF", "OURSELVES", "YOURSELVES", "THEMSELVES", "ONESELF", "THYSELF", "MESELF", "YERSELF"]},
            {"subgroup": "demonstratives", "words": ["THIS", "THAT", "THESE", "THOSE", "HERE", "THERE", "SUCH", "SAME", "FORMER", "LATTER", "YONDER", "YON"]},
            {"subgroup": "interrogatives", "words": ["WHO", "WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHICH", "WHOSE", "WHOM", "WHETHER", "WHATEVER", "WHEREVER"]},
            {"subgroup": "determiners", "words": ["THE", "A", "AN", "SOME", "ANY", "EACH", "EVERY", "NO", "ALL", "BOTH", "FEW", "MANY"]},
            {"subgroup": "auxiliary verbs", "words": ["BE", "HAVE", "DO", "WILL", "SHALL", "WOULD", "SHOULD", "MAY", "MIGHT", "CAN", "COULD", "MUST"]},
            {"subgroup": "adverbs of frequency", "words": ["ALWAYS", "USUALLY", "OFTEN", "SOMETIMES", "OCCASIONALLY", "RARELY", "SELDOM", "NEVER", "FREQUENTLY", "REGULARLY", "NORMALLY", "CONSTANTLY"]},
            {"subgroup": "indefinite pronouns", "words": ["SOMEONE", "ANYONE", "EVERYONE", "NOBODY", "SOMETHING", "ANYTHING", "EVERYTHING", "NOTHING", "SOMEBODY", "ANYBODY", "EVERYBODY", "NONE"]},
            {"subgroup": "comparative adjectives", "words": ["BETTER", "WORSE", "BIGGER", "SMALLER", "FASTER", "SLOWER", "HIGHER", "LOWER", "STRONGER", "WEAKER", "OLDER", "YOUNGER"]}
        ]
    },

    "WORDPLAY_DOUBLE_MEANING": {
        "description": "polysemy/ambiguity: different senses, shared nicknames, contronyms",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "bank meanings", "words": ["RIVER", "FINANCIAL", "SAVINGS", "SLOPE", "TURN", "RELY", "STORE", "ROW", "TIER", "HEAP", "MASS", "RIDGE"]},
            {"subgroup": "bark meanings", "words": ["TREE", "DOG", "SHIP", "SOUND", "COVER", "SKIN", "SHOUT", "ORDER", "COMMAND", "ROUGH", "HARSH", "GRUFF"]},
            {"subgroup": "spring meanings", "words": ["SEASON", "WATER", "COIL", "JUMP", "LEAP", "BOUNCE", "ELASTIC", "SOURCE", "ORIGIN", "YOUTH", "ENERGY", "FRESH"]},
            {"subgroup": "mine meanings", "words": ["EXCAVATION", "EXPLOSIVE", "POSSESS", "BELONGING", "TUNNEL", "PIT", "QUARRY", "DIG", "EXTRACT", "WEALTH", "RESOURCE", "CLAIM"]},
            {"subgroup": "match meanings", "words": ["FIRE", "GAME", "PAIR", "EQUAL", "CONTEST", "COMPETITION", "SUITABLE", "CORRESPOND", "AGREE", "RIVAL", "COMPLEMENT", "FIT"]},
            {"subgroup": "bat meanings", "words": ["ANIMAL", "SPORTS", "CLUB", "STICK", "HIT", "STRIKE", "SWING", "FLUTTER", "WINK", "BLINK", "NOCTURNAL", "CRICKET"]},
            {"subgroup": "seal meanings", "words": ["ANIMAL", "STAMP", "CLOSE", "SECURE", "EMBLEM", "MARK", "WATERTIGHT", "CONFIRM", "GUARANTEE", "MARINE", "WAX", "OFFICIAL"]},
            {"subgroup": "wave meanings", "words": ["OCEAN", "HAND", "HAIR", "PHYSICS", "GREETING", "MOTION", "SURGE", "TREND", "PATTERN", "SIGNAL", "GOODBYE", "FLUTTER"]},
            {"subgroup": "fair meanings", "words": ["JUST", "CARNIVAL", "WEATHER", "LIGHT", "BEAUTIFUL", "AVERAGE", "BLONDE", "IMPARTIAL", "REASONABLE", "FESTIVAL", "MARKET", "EXHIBIT"]},
            {"subgroup": "light meanings", "words": ["BRIGHT", "WEIGHT", "IGNITE", "LAMP", "COLOR", "EASY", "GENTLE", "KNOWLEDGE", "DAWN", "PALE", "TRIVIAL", "AGILE"]},
            {"subgroup": "ring meanings", "words": ["JEWELRY", "CIRCLE", "SOUND", "BELL", "CALL", "ARENA", "GROUP", "SURROUND", "ECHO", "RESONANCE", "WEDDING", "BOXING"]},
            {"subgroup": "rose meanings", "words": ["FLOWER", "STOOD", "COLOR", "INCREASE", "ASCEND", "PINK", "CLIMBED", "WINE", "PERFUME", "BLUSH", "DAWN", "ELEVATED"]}
        ]
    },

    "TEMPORAL_SEQUENTIAL": {
        "description": "ordered/cyclical series where rank or sequence essential",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "months", "words": ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY", "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"]},
            {"subgroup": "days of week", "words": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY", "WEEKDAY", "WEEKEND", "TODAY", "TOMORROW", "YESTERDAY"]},
            {"subgroup": "seasons", "words": ["SPRING", "SUMMER", "FALL", "AUTUMN", "WINTER", "SEASONAL", "QUARTERLY", "ANNUAL", "PERENNIAL", "EQUINOX", "SOLSTICE", "HARVEST"]},
            {"subgroup": "meal times", "words": ["BREAKFAST", "BRUNCH", "LUNCH", "TEA", "DINNER", "SUPPER", "SNACK", "APPETIZER", "DESSERT", "MIDNIGHT", "AFTERNOON", "EVENING"]},
            {"subgroup": "life stages", "words": ["INFANT", "TODDLER", "CHILD", "TEENAGER", "ADULT", "MIDDLE", "ELDERLY", "SENIOR", "YOUTH", "ADOLESCENT", "MATURE", "AGED"]},
            {"subgroup": "decades", "words": ["TWENTIES", "THIRTIES", "FORTIES", "FIFTIES", "SIXTIES", "SEVENTIES", "EIGHTIES", "NINETIES", "MILLENNIUM", "CENTURY", "DECADE", "ERA"]},
            {"subgroup": "time of day", "words": ["DAWN", "MORNING", "NOON", "AFTERNOON", "EVENING", "DUSK", "NIGHT", "MIDNIGHT", "SUNRISE", "SUNSET", "TWILIGHT", "DAYBREAK"]},
            {"subgroup": "academic grades", "words": ["KINDERGARTEN", "FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH", "FRESHMAN", "SOPHOMORE", "JUNIOR"]},
            {"subgroup": "military ranks", "words": ["PRIVATE", "CORPORAL", "SERGEANT", "LIEUTENANT", "CAPTAIN", "MAJOR", "COLONEL", "GENERAL", "ADMIRAL", "MARSHAL", "COMMANDER", "OFFICER"]},
            {"subgroup": "book parts", "words": ["PREQUEL", "PREFACE", "FOREWORD", "INTRODUCTION", "SEQUEL", "TRILOGY", "SERIES", "VOLUME", "CHAPTER", "PROLOGUE", "EPILOGUE", "APPENDIX"]},
            {"subgroup": "geologic periods", "words": ["CAMBRIAN", "ORDOVICIAN", "SILURIAN", "DEVONIAN", "CARBONIFEROUS", "PERMIAN", "TRIASSIC", "JURASSIC", "CRETACEOUS", "PALEOGENE", "NEOGENE", "QUATERNARY"]},
            {"subgroup": "project phases", "words": ["PLANNING", "DESIGN", "DEVELOPMENT", "TESTING", "IMPLEMENTATION", "DEPLOYMENT", "MAINTENANCE", "REVIEW", "CLOSURE", "INITIATION", "EXECUTION", "MONITORING"]}
        ]
    },

    "NUMERICAL_QUANTITATIVE": {
        "description": "patterns with digits, counts, arithmetic, measurement, numeric symbols",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "roman numerals", "words": ["I", "V", "X", "L", "C", "D", "M", "IV", "IX", "XL", "XC", "CD"]},
            {"subgroup": "number prefixes", "words": ["UNI", "BI", "TRI", "QUAD", "PENTA", "HEXA", "SEPTA", "OCTA", "NONA", "DECA", "CENTI", "MILLI"]},
            {"subgroup": "fractions", "words": ["HALF", "THIRD", "QUARTER", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH", "NINTH", "TENTH", "HUNDREDTH", "THOUSANDTH", "WHOLE"]},
            {"subgroup": "group size terms", "words": ["DOZEN", "GROSS", "SCORE", "PAIR", "COUPLE", "TRIO", "QUARTET", "QUINTET", "SEXTET", "SEPTET", "OCTET", "NONET"]},
            {"subgroup": "metric units", "words": ["METER", "GRAM", "LITER", "SECOND", "AMPERE", "KELVIN", "MOLE", "CANDELA", "HERTZ", "NEWTON", "PASCAL", "JOULE"]},
            {"subgroup": "binary terms", "words": ["BIT", "BYTE", "KILOBYTE", "MEGABYTE", "GIGABYTE", "TERABYTE", "BINARY", "DIGITAL", "ANALOG", "HEXADECIMAL", "OCTAL", "DECIMAL"]},
            {"subgroup": "angles", "words": ["ACUTE", "RIGHT", "OBTUSE", "STRAIGHT", "REFLEX", "COMPLEMENTARY", "SUPPLEMENTARY", "VERTICAL", "ADJACENT", "CORRESPONDING", "ALTERNATE", "INTERIOR"]},
            {"subgroup": "prime numbers", "words": ["TWO", "THREE", "FIVE", "SEVEN", "ELEVEN", "THIRTEEN", "SEVENTEEN", "NINETEEN", "TWENTY-THREE", "TWENTY-NINE", "THIRTY-ONE", "THIRTY-SEVEN"]},
            {"subgroup": "counting by tens", "words": ["TEN", "TWENTY", "THIRTY", "FORTY", "FIFTY", "SIXTY", "SEVENTY", "EIGHTY", "NINETY", "HUNDRED", "THOUSAND", "MILLION"]},
            {"subgroup": "percentage terms", "words": ["PERCENT", "QUARTER", "HALF", "THIRD", "WHOLE", "FRACTION", "DECIMAL", "RATIO", "PROPORTION", "RATE", "YIELD", "SHARE"]},
            {"subgroup": "mathematical operations", "words": ["ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "SQUARE", "CUBE", "ROOT", "POWER", "FACTORIAL", "LOGARITHM", "EXPONENT", "DERIVATIVE"]},
            {"subgroup": "ordinal numbers", "words": ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH", "EIGHTH", "NINTH", "TENTH", "ELEVENTH", "TWELFTH"]}
        ]
    },

    "LEXICAL_ETYMOLOGY": {
        "description": "shared origin: loanwords, eponyms, language tokens",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "french loanwords", "words": ["RESTAURANT", "CAFE", "MENU", "CUISINE", "CHEF", "GOURMET", "BOUQUET", "BOUTIQUE", "BUREAU", "DEPOT", "GARAGE", "BALLET"]},
            {"subgroup": "italian loanwords", "words": ["PIZZA", "PASTA", "OPERA", "PIANO", "SOPRANO", "ALTO", "TEMPO", "VILLA", "STUDIO", "UMBRELLA", "VOLCANO", "GRAFFITI"]},
            {"subgroup": "spanish loanwords", "words": ["PLAZA", "PATIO", "SIESTA", "FIESTA", "SALSA", "TACO", "BURRITO", "ADOBE", "CANYON", "RODEO", "LASSO", "BRONCO"]},
            {"subgroup": "japanese loanwords", "words": ["SUSHI", "SAKE", "KARAOKE", "ORIGAMI", "TSUNAMI", "KARATE", "NINJA", "SAMURAI", "KIMONO", "BONSAI", "EMOJI", "MANGA"]},
            {"subgroup": "german loanwords", "words": ["KINDERGARTEN", "HAMBURGER", "PRETZEL", "SAUERKRAUT", "BRATWURST", "DACHSHUND", "WANDERLUST", "ZEITGEIST", "ANGST", "UBER", "FEST", "STEIN"]},
            {"subgroup": "arabic loanwords", "words": ["ALGEBRA", "ALGORITHM", "ZERO", "COFFEE", "SUGAR", "COTTON", "MAGAZINE", "ADMIRAL", "ARSENAL", "GAZELLE", "GIRAFFE", "SAFARI"]},
            {"subgroup": "greek roots", "words": ["PHILOSOPHY", "DEMOCRACY", "BIOLOGY", "TECHNOLOGY", "PSYCHOLOGY", "GEOGRAPHY", "MATHEMATICS", "PHYSICS", "POLITICS", "ECONOMICS", "ATHLETICS", "SYMPHONY"]},
            {"subgroup": "latin roots", "words": ["DOCTOR", "HOSPITAL", "MEDICINE", "PATIENT", "VIRUS", "VACCINE", "SCIENCE", "LABORATORY", "EXPERIMENT", "UNIVERSITY", "EDUCATION", "STUDENT"]},
            {"subgroup": "native american loanwords", "words": ["CANOE", "KAYAK", "TOBACCO", "TOMATO", "POTATO", "CHOCOLATE", "COYOTE", "HURRICANE", "BARBECUE", "HAMMOCK", "IGUANA", "MAIZE"]},
            {"subgroup": "dutch loanwords", "words": ["COOKIE", "YACHT", "DOCK", "DECK", "CRUISE", "FREIGHT", "LANDSCAPE", "SKETCH", "EASEL", "BRANDY", "COLESLAW", "WAFFLE"]},
            {"subgroup": "hindi loanwords", "words": ["YOGA", "KARMA", "MANTRA", "GURU", "AVATAR", "JUNGLE", "SHAMPOO", "PAJAMAS", "BUNGALOW", "CHUTNEY", "CURRY", "KHAKI"]},
            {"subgroup": "russian loanwords", "words": ["VODKA", "CZAR", "TSAR", "MAMMOTH", "SPUTNIK", "COSMONAUT", "GLASNOST", "PERESTROIKA", "TROIKA", "DACHA", "POGROM", "GULAG"]}
        ]
    },

    "SOCIOLINGUISTIC_REGISTER": {
        "description": "grouping by dialect, region, era, register (slang, formal, jargon)",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "british vs american", "words": ["LIFT", "ELEVATOR", "LORRY", "TRUCK", "BOOT", "TRUNK", "BONNET", "HOOD", "PETROL", "GAS", "FLAT", "APARTMENT"]},
            {"subgroup": "medical terminology", "words": ["DIAGNOSIS", "PROGNOSIS", "SYMPTOM", "SYNDROME", "CHRONIC", "ACUTE", "BENIGN", "MALIGNANT", "THERAPY", "TREATMENT", "PRESCRIPTION", "DOSAGE"]},
            {"subgroup": "legal jargon", "words": ["PLAINTIFF", "DEFENDANT", "LITIGATION", "JURISDICTION", "PRECEDENT", "STATUTE", "TESTIMONY", "EVIDENCE", "VERDICT", "ACQUITTAL", "CONVICTION", "APPEAL"]},
            {"subgroup": "internet slang", "words": ["LOL", "LMAO", "ROFL", "BRB", "AFK", "IMO", "TBH", "SMH", "FOMO", "YOLO", "SELFIE", "HASHTAG"]},
            {"subgroup": "business buzzwords", "words": ["SYNERGY", "LEVERAGE", "PARADIGM", "DISRUPT", "INNOVATE", "PIVOT", "SCALABLE", "AGILE", "BANDWIDTH", "ECOSYSTEM", "STAKEHOLDER", "DELIVERABLE"]},
            {"subgroup": "academic terms", "words": ["THESIS", "HYPOTHESIS", "METHODOLOGY", "ANALYSIS", "SYNTHESIS", "LITERATURE", "CITATION", "BIBLIOGRAPHY", "ABSTRACT", "CONCLUSION", "DISSERTATION", "PEER"]},
            {"subgroup": "gen z slang", "words": ["SLAY", "PERIODT", "BUSSIN", "DRIP", "FLEX", "VIBE", "STAN", "SIMP", "SALTY", "GHOSTING", "CANCELLED", "WOKE"]},
            {"subgroup": "sports terminology", "words": ["OFFENSE", "DEFENSE", "TOUCHDOWN", "HOMERUN", "STRIKEOUT", "PENALTY", "FOUL", "TIMEOUT", "OVERTIME", "PLAYOFF", "CHAMPIONSHIP", "TOURNAMENT"]},
            {"subgroup": "culinary terms", "words": ["JULIENNE", "BRUNOISE", "SAUTE", "BLANCH", "BRAISE", "DEGLAZE", "EMULSIFY", "FLAMBE", "POACH", "REDUCTION", "ROUX", "ZEST"]},
            {"subgroup": "music terminology", "words": ["TEMPO", "RHYTHM", "MELODY", "HARMONY", "PITCH", "TIMBRE", "DYNAMICS", "CRESCENDO", "DIMINUENDO", "STACCATO", "LEGATO", "FORTE"]},
            {"subgroup": "nautical terms", "words": ["PORT", "STARBOARD", "BOW", "STERN", "AFT", "FORE", "GALLEY", "BERTH", "KNOT", "FATHOM", "LEAGUE", "NAUTICAL"]},
            {"subgroup": "military jargon", "words": ["ROGER", "COPY", "AFFIRMATIVE", "NEGATIVE", "ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT", "GOLF", "HOTEL"]}
        ]
    },

    "CROSS_LINGUISTIC": {
        "description": "translations of same concept across languages",
        "has_subgroups": True,
        "examples": [
            {"subgroup": "hello in languages", "words": ["HELLO", "HOLA", "BONJOUR", "CIAO", "GUTEN", "KONNICHIWA", "NAMASTE", "SALAAM", "SHALOM", "ALOHA", "JAMBO", "SAWADEE"]},
            {"subgroup": "goodbye in languages", "words": ["GOODBYE", "ADIOS", "AUREVOIR", "ARRIVEDERCI", "AUFWIEDERSEHEN", "SAYONARA", "FAREWELL", "ADIEU", "CHEERIO", "CIAO", "VALE", "ALOHA"]},
            {"subgroup": "thank you languages", "words": ["THANKS", "GRACIAS", "MERCI", "DANKE", "GRAZIE", "ARIGATO", "SPASIBO", "SHUKRAN", "OBRIGADO", "KIITOS", "TAKK", "EFHARISTO"]},
            {"subgroup": "colors in spanish", "words": ["ROJO", "AZUL", "VERDE", "AMARILLO", "NEGRO", "BLANCO", "GRIS", "MARRON", "ROSA", "NARANJA", "MORADO", "DORADO"]},
            {"subgroup": "numbers in french", "words": ["UN", "DEUX", "TROIS", "QUATRE", "CINQ", "SIX", "SEPT", "HUIT", "NEUF", "DIX", "VINGT", "CENT"]},
            {"subgroup": "animals in german", "words": ["HUND", "KATZE", "PFERD", "KUH", "SCHWEIN", "SCHAF", "ZIEGE", "HUHN", "ENTE", "GANS", "MAUS", "RATTE"]},
            {"subgroup": "food in italian", "words": ["PANE", "FORMAGGIO", "VINO", "ACQUA", "CARNE", "PESCE", "FRUTTA", "VERDURA", "LATTE", "BURRO", "OLIO", "SALE"]},
            {"subgroup": "family in japanese", "words": ["CHICHI", "HAHA", "ANI", "ANE", "OTOTO", "IMOTO", "SOFU", "SOBO", "OJI", "OBA", "ITOKO", "KAZOKU"]},
            {"subgroup": "days in latin", "words": ["DIES", "SOLIS", "LUNAE", "MARTIS", "MERCURII", "IOVIS", "VENERIS", "SATURNI", "HEBDOMAS", "MENSIS", "ANNUS", "SAECULUM"]},
            {"subgroup": "body parts chinese", "words": ["TOU", "SHOU", "JIAO", "YAN", "BI", "ZUI", "ER", "XIN", "FEI", "WEI", "GAN", "SHEN"]},
            {"subgroup": "weather in russian", "words": ["POGODA", "SOLNTSE", "DOZHD", "SNEG", "VETER", "OBLAKO", "GROZA", "TUMAN", "GRAD", "RADUGA", "MOLNIYA", "GROM"]},
            {"subgroup": "love in languages", "words": ["LOVE", "AMOR", "AMOUR", "LIEBE", "AMORE", "AI", "LYUBOV", "AGAPE", "ISHQ", "PREM", "CINTA", "SEVGI"]}
        ]
    }
}

class CategoricalPuzzleGenerator:
    """generator for nyt connections style puzzles"""

    def __init__(self, patterns: Dict = None):
        self.patterns = patterns or CATEGORICAL_PATTERNS
        self.used_patterns = []
        self.puzzle_history = []

    def get_words_from_pattern(self, pattern_name: str, index: int = None) -> Tuple[List[str], str]:
        """get words from pattern with subgroup"""
        pattern = self.patterns[pattern_name]

        if pattern.get("has_subgroups", False):
            if index is None:
                index = random.randint(0, len(pattern["examples"]) - 1)
            example = pattern["examples"][index % len(pattern["examples"])]
            words = example["words"]
            subgroup = example.get("subgroup", pattern_name)
            return words, subgroup
        else:
            example = random.choice(pattern["examples"])
            if isinstance(example, dict):
                words = example["words"]
            else:
                words = example
            return words, pattern_name

    def generate_4_1_complex(self) -> Dict:
        """generate 4:1 odd one out"""
        patterns = random.sample(list(self.patterns.keys()), 2)
        main_pattern = patterns[0]
        outlier_pattern = patterns[1]

        main_words, main_subgroup = self.get_words_from_pattern(main_pattern)
        if len(main_words) < 4:
            return None
        main_words = random.sample(main_words, min(4, len(main_words)))

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
        """generate 5:2 pattern"""
        patterns = random.sample(list(self.patterns.keys()), 2)
        main_pattern = patterns[0]
        minor_pattern = patterns[1]

        main_words, main_subgroup = self.get_words_from_pattern(main_pattern)
        minor_words, minor_subgroup = self.get_words_from_pattern(minor_pattern)

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
        """generate 7:3 pattern"""
        patterns = random.sample(list(self.patterns.keys()), 2)
        main_pattern = patterns[0]
        minor_pattern = patterns[1]

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
        """generate 8:2:2 three group pattern"""
        patterns = random.sample(list(self.patterns.keys()), 3)

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
            "explanation": f"Group 1: {main_subgroup}, Group 2: {minor1_subgroup}, Group 3: {minor2_subgroup}"
        }

    def generate_10_3_3_complex(self) -> Dict:
        """generate 10:3:3 three group pattern"""
        patterns = random.sample(list(self.patterns.keys()), 3)

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
            "explanation": f"Group 1: {main_subgroup}, Group 2: {minor1_subgroup}, Group 3: {minor2_subgroup}"
        }

    def generate_complex_examples(self, num_per_pattern: Dict[str, int]) -> List[Dict]:
        """generate examples with distribution"""
        all_examples = []

        for pattern_type, count in num_per_pattern.items():
            print(f"\ngenerating {count} examples of pattern {pattern_type}")

            pattern_examples = []
            attempts = 0
            max_attempts = count * 10

            while len(pattern_examples) < count and attempts < max_attempts:
                attempts += 1

                if len(pattern_examples) % 20 == 0 and len(pattern_examples) > 0:
                    print(f"  generated {len(pattern_examples)}/{count}")

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
                print(f"  warning: only generated {len(pattern_examples)}/{count} after {attempts} attempts")

            print(f"  completed: {len(pattern_examples)} examples")

        return all_examples

    def main(self):
        """main generation function"""
        print("="*60)
        print("categorical linguistic reasoning dataset generator")
        print("nyt connections patterns from gpt-5 analysis")
        print("="*60)

        Path('data/output').mkdir(parents=True, exist_ok=True)

        # distribution
        distribution = {
            "4:1": 100,
            "5:2": 150,
            "7:3": 150,
            "8:2:2": 200,
            "10:3:3": 200
        }

        print("\ntarget distribution:")
        for pattern, count in distribution.items():
            print(f"  {pattern}: {count} examples")
        print(f"  total: {sum(distribution.values())} examples")

        print(f"\nusing {len(self.patterns)} pattern types from nyt analysis")
        print("pattern types:", ", ".join(list(self.patterns.keys())[:5]) + "...")

        # generate
        all_examples = self.generate_complex_examples(distribution)
        all_examples = [e for e in all_examples if e is not None]

        print(f"\nsuccessfully generated: {len(all_examples)} examples")

        # save
        dataset = {
            "task": "categorical_linguistic_reasoning",
            "description": "nyt connections patterns from gpt-5 taxonomy analysis",
            "total_examples": len(all_examples),
            "pattern_types": list(self.patterns.keys()),
            "examples": all_examples
        }

        with open('data/output/preconn_categorical_raw.json', 'w') as f:
            json.dump(dataset, f, indent=2)

        # jsonl for training
        with open('data/output/preconn_categorical_raw.jsonl', 'w') as f:
            for ex in all_examples:
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

        print("\nfiles created:")
        print("  - data/output/preconn_categorical_raw.json")
        print("  - data/output/preconn_categorical_raw.jsonl")

        # stats
        print("\npattern usage (with subgroups):")
        pattern_counts = {}
        for ex in all_examples:
            exp = ex.get("explanation", "")
            parts = exp.split(", ")
            for part in parts:
                if ":" in part:
                    _, subgroup = part.split(": ", 1)
                    pattern_counts[subgroup] = pattern_counts.get(subgroup, 0) + 1

        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  {pattern}: {count} uses")

        # samples
        print("\nsample examples:")
        for i in range(min(5, len(all_examples))):
            ex = all_examples[i]
            print(f"\nexample {i+1} ({ex['pattern']}):")
            print(f"  pattern: {ex.get('explanation', 'N/A')}")
            print(f"  input: {ex['input'][:100]}...")

if __name__ == "__main__":
    generator = CategoricalPuzzleGenerator()
    generator.main()