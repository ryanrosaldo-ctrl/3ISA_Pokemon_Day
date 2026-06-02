import json
import concurrent.futures
import urllib.request
import urllib.error
import time

# Region mappings by Gen/ID range
def get_region_and_gen(pkmn_id):
    if 1 <= pkmn_id <= 151:
        return "Kanto", 1
    elif 152 <= pkmn_id <= 251:
        return "Johto", 2
    elif 252 <= pkmn_id <= 386:
        return "Hoenn", 3
    elif 387 <= pkmn_id <= 493:
        return "Sinnoh", 4
    elif 494 <= pkmn_id <= 649:
        return "Unova", 5
    elif 650 <= pkmn_id <= 721:
        return "Kalos", 6
    elif 722 <= pkmn_id <= 809:
        return "Alola", 7
    elif 810 <= pkmn_id <= 898:
        return "Galar", 8
    elif 899 <= pkmn_id <= 905:
        # Hisuian/Legends Arceus (usually counted under Sinnoh or Gen 8)
        return "Sinnoh", 8
    elif 906 <= pkmn_id <= 1025:
        return "Paldea", 9
    else:
        return "Unknown", 0

# List of Paradox Pokemon names to identify restriction
PARADOX_POKEMON = {
    "great-tusk", "scream-tail", "brute-bonnet", "flutter-mane", "slither-wing",
    "sandy-shocks", "iron-treads", "iron-bundle", "iron-hands", "iron-jugulis",
    "iron-moth", "iron-thorns", "roaring-moon", "iron-valiant", "walking-wake",
    "iron-leaves", "gouging-fire", "raging-bolt", "iron-boulder", "iron-crown",
    "koraidon", "miraidon" # Cover legendaries/paradoxes
}

def fetch_pokemon_data(pkmn_id):
    url_poke = f"https://pokeapi.co/api/v2/pokemon/{pkmn_id}"
    url_spec = f"https://pokeapi.co/api/v2/pokemon-species/{pkmn_id}"
    
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        # Fetch basic info
        req_poke = urllib.request.Request(url_poke, headers=headers)
        with urllib.request.urlopen(req_poke, timeout=10) as response:
            data = json.loads(response.read().decode())
            
        # Fetch species info
        req_spec = urllib.request.Request(url_spec, headers=headers)
        with urllib.request.urlopen(req_spec, timeout=10) as response:
            spec_data = json.loads(response.read().decode())
            
        region, gen = get_region_and_gen(pkmn_id)
        
        # Parse types
        type_1 = data["types"][0]["type"]["name"].capitalize()
        type_2 = data["types"][1]["type"]["name"].capitalize() if len(data["types"]) > 1 else ""
        
        # Parse stats
        stats = {s["stat"]["name"]: s["base_stat"] for s in data["stats"]}
        hp = stats.get("hp", 0)
        attack = stats.get("attack", 0)
        defense = stats.get("defense", 0)
        special_attack = stats.get("special-attack", 0)
        special_defense = stats.get("special-defense", 0)
        speed = stats.get("speed", 0)
        
        # Determine restricted status
        is_legendary = spec_data.get("is_legendary", False)
        is_mythical = spec_data.get("is_mythical", False)
        api_name = data["name"].lower()
        is_restricted = is_legendary or is_mythical or (api_name in PARADOX_POKEMON)
        
        # Capitalize name nicely
        name = data["name"].replace("-", " ").title()
        
        pokemon_entry = {
            "id": pkmn_id,
            "name": name,
            "api_name": api_name,
            "native_region": region,
            "generation": gen,
            "type_1": type_1,
            "type_2": type_2,
            "hp": hp,
            "attack": attack,
            "defense": defense,
            "special_attack": special_attack,
            "special_defense": special_defense,
            "speed": speed,
            "is_restricted": is_restricted
        }
        print(f"[{pkmn_id}] Fetched {name} ({region}) successfully.")
        return pokemon_entry
        
    except Exception as e:
        print(f"Error fetching ID {pkmn_id}: {e}")
        return None

def main():
    print("Starting concurrent PokéAPI download for all 1025 Pokémon...")
    start_time = time.time()
    
    pokemon_list = []
    
    # Run in parallel using 40 worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
        futures = {executor.submit(fetch_pokemon_data, i): i for i in range(1, 1026)}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                pokemon_list.append(res)
                
    # Sort by ID
    pokemon_list.sort(key=lambda x: x["id"])
    
    # Save to JSON
    with open("pokemon_data.json", "w") as f:
        json.dump(pokemon_list, f, indent=2)
        
    end_time = time.time()
    print(f"Finished downloading {len(pokemon_list)} Pokémon in {end_time - start_time:.2f} seconds!")

if __name__ == "__main__":
    main()
