import json
from pathlib import Path

def create_manifest():
    manifest = []
    
    categories = {
        "LETTERS": [
            ("A", "letter, vowel"), ("B", "letter, plosive"), ("C", "letter, sibilant"),
            ("D", "letter, plosive"), ("E", "letter, vowel"), ("F", "letter, fricative"),
            ("G", "letter, plosive"), ("H", "letter, aspirate"), ("I", "letter, vowel"),
            ("J", "letter, affricate"), ("K", "letter, plosive"), ("L", "letter, liquid"),
            ("M", "letter, nasal"), ("N", "letter, nasal"), ("O", "letter, vowel"),
            ("P", "letter, plosive"), ("Q", "letter, plosive"), ("R", "letter, liquid"),
            ("S", "letter, sibilant"), ("T", "letter, plosive"), ("U", "letter, vowel"),
            ("V", "letter, fricative"), ("W", "letter, glide"), ("X", "letter, sibilant"),
            ("Y", "letter, glide"), ("Z", "letter, sibilant")
        ],
        "DIGITS": [
            ("0", "digit, sibilant"), ("1", "digit, nasal"), ("2", "digit, vowel"),
            ("3", "digit, fricative"), ("4", "digit, liquid"), ("5", "digit, diphthong"),
            ("6", "digit, sibilant"), ("7", "digit, fricative"), ("8", "digit, diphthong"),
            ("9", "digit, nasal")
        ],
        "PUNCTUATION": [
            ("comma", "nasal, vowel"), ("period", "plosive, high-vowel"), ("question mark", "velar, nasal"),
            ("exclamation mark", "nasal, sibilant"), ("colon", "velar, alveolar"), ("semicolon", "sibilant, nasal"),
            ("dash", "alveolar, fricative"), ("hyphen", "glottal, labial"), ("apostrophe", "alveolar, fricative"),
            ("quote", "velar, alveolar"), ("slash", "sibilant, lateral"), ("backslash", "bilabial, sibilant"),
            ("at", "vowel, plosive"), ("number", "nasal, labial"), ("percent", "plosive, sibilant"),
            ("ampersand", "vowel, nasal, plosive"), ("plus", "plosive, sibilant"), ("equals", "vowel, sibilant")
        ],
        "UI_NAVIGATION": [
            ("button", "bilabial, alveolar, nasal"), ("link", "lateral, velar, nasal"), ("list", "lateral, sibilant, plosive"),
            ("list item", "lateral, sibilant, nasal"), ("heading", "glottal, nasal, velar"), ("checkbox", "affricate, velar, sibilant"),
            ("checked", "affricate, velar, plosive"), ("unchecked", "nasal, affricate, velar"), ("selected", "sibilant, lateral, plosive"),
            ("not selected", "nasal, lateral, plosive"), ("expanded", "vowel, plosive, nasal"), ("collapsed", "velar, lateral, sibilant"),
            ("unavailable", "vowel, nasal, lateral"), ("edit", "vowel, alveolar, plosive"), ("editable", "vowel, alveolar, lateral"),
            ("menu", "nasal, vowel"), ("menu item", "nasal, vowel, alveolar"), ("dialog", "alveolar, velar, nasal"),
            ("tab", "alveolar, bilabial"), ("table", "alveolar, bilabial, lateral"), ("row", "liquid, vowel"),
            ("column", "velar, lateral, nasal"), ("tree view", "alveolar, fricative, vowel"), ("progress bar", "plosive, liquid, sibilant"),
            ("slider", "sibilant, lateral, liquid"), ("radio button", "liquid, alveolar, plosive"), ("visited", "fricative, alveolar, plosive"),
            ("blank", "bilabial, lateral, velar"), ("space", "sibilant, bilabial, velar"), ("capital", "velar, bilabial, lateral"),
            ("level one", "lateral, labial, nasal"), ("level two", "lateral, labial, vowel"), ("pressed", "plosive, liquid, sibilant"),
            ("not pressed", "nasal, liquid, sibilant"), ("frame", "fricative, liquid, nasal"), ("status bar", "sibilant, alveolar, liquid"),
            ("toolbar", "alveolar, liquid, bilabial"), ("window", "glide, nasal, alveolar"), ("alert", "vowel, liquid, alveolar"),
            ("notification", "nasal, fricative, alveolar")
        ],
        "SHORT_WORDS": [
            ("pop", "plosive, aspirated"), ("dog", "plosive, voiced"), ("cat", "plosive, voiceless"),
            ("noon", "nasal, continuous"), ("sing", "velar, nasal, continuous"), ("shush", "fricative, voiceless"),
            ("laugh", "fricative, labiodental"), ("thin", "fricative, interdental"), ("zip", "sibilant, voiced"),
            ("fizz", "sibilant, continuous"), ("roll", "liquid, lateral"), ("lily", "liquid, alveolar"),
            ("yawn", "approximant, nasal"), ("wet", "approximant, plosive"), ("area", "vowel-heavy"),
            ("idea", "vowel-heavy"), ("blast", "initial-consonant cluster"), ("split", "initial-consonant cluster"),
            ("grasp", "final-consonant cluster"), ("F", "historical_problem, fricative"), ("N", "historical_problem, nasal"),
            ("m", "historical_problem, nasal"), ("b", "historical_problem, plosive"), ("V", "historical_problem, fricative"),
            ("list", "historical_problem, cluster"), ("link", "historical_problem, velar"), ("comma", "historical_problem, nasal")
        ],
        "PHRASES": [
            ("Save button", "UI phrase, sibilant, plosive"), ("Search edit", "UI phrase, sibilant, plosive"),
            ("Heading level two", "UI phrase, nasal, lateral"), ("Link visited", "UI phrase, lateral, fricative"),
            ("Checkbox checked", "UI phrase, affricate, sibilant"), ("Menu collapsed", "UI phrase, nasal, sibilant"),
            ("Dialog unavailable", "UI phrase, alveolar, lateral"), ("List with five items", "UI phrase, lateral, labial"),
            ("Row three column two", "UI phrase, liquid, velar"), ("Selected tab", "UI phrase, sibilant, alveolar"),
            ("Download complete", "UI phrase, alveolar, velar"), ("Tree view expanded", "UI phrase, alveolar, fricative"),
            ("Radio button checked", "UI phrase, liquid, affricate"), ("Editable text", "UI phrase, vowel, sibilant"),
            ("Table with four rows", "UI phrase, labial, liquid"), ("Capital A", "UI phrase, velar, vowel"),
            ("Window focused", "UI phrase, glide, fricative"), ("Progress bar fifty percent", "UI phrase, liquid, sibilant"),
            ("Toolbar expanded", "UI phrase, liquid, vowel"), ("Alert notification", "UI phrase, liquid, nasal")
        ],
        "SENTENCES": [
            ("The selected button is unavailable.", "sentence control, UI notification"),
            ("Your changes have been saved.", "sentence control, status notification"),
            ("Are you sure you want to exit?", "sentence control, question dialog"),
            ("An error occurred while loading the page.", "sentence control, error alert"),
            ("The download has completed successfully.", "sentence control, status alert"),
            ("Please select an option from the menu.", "sentence control, instructions"),
            ("The document has been updated.", "sentence control, confirmation"),
            ("A new notification is available.", "sentence control, status alert"),
            ("Press tab to navigate to the next field.", "sentence control, instructions"),
            ("The application is starting up.", "sentence control, status alert")
        ]
    }
    
    item_idx = 1
    for cat_name, items in categories.items():
        for item in items:
            text = item[0]
            tags = item[1]
            is_historical = "historical_problem" in tags or text in ["F", "N", "m", "b", "V", "list", "link", "comma"]
            
            manifest.append({
                "item_id": f"P2AT_{item_idx:03d}",
                "text": text,
                "category": cat_name,
                "historical_problem_item": is_historical,
                "phonetic_coverage_tags": [tag.strip() for tag in tags.split(",")]
            })
            item_idx += 1
            
    out_path = Path("C:/projects/piper-screen-reader-research/training/results/phase2at/phase2at-corpus-manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Phase 2AT Corpus Manifest generated successfully at {out_path} ({len(manifest)} items).")

if __name__ == "__main__":
    create_manifest()
