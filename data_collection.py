import lyricsgenius
import os

# config
GENIUS_ACCESS_TOKEN = 'vHF_WTIZAk6saoLtXsXAyh7lxKaCLcKJwiqm_88gdBiXqNVIPCyVTL33KNkTdmbh'
ARTIST_NAME = "Tede"
OUTPUT_FILENAME = "assets/input/input2.txt"

def fetch_and_save_lyrics(token: str, artist_name: str, output_file: str):
    """
    Connects to Genius API, fetches all songs for the specified artist,
    and saves the lyrics to a single text file.
    """
    try:
        # Initialize Genius API client with increased timeout and retries
        genius = lyricsgenius.Genius(token, timeout=20, retries=3)
        
        # Enable verbose output to track progress
        genius.verbose = True
        # Optional: remove section headers like [Verse], [Chorus]
        genius.remove_section_headers = True 

        print(f"[INFO] Searching for artist: {artist_name}...")
        
        # Search for the artist and download all songs
        artist = genius.search_artist(artist_name, max_songs=None, sort="title", include_features=False)

        if not artist:
            print(f"[ERROR] Artist '{artist_name}' not found.")
            return

        print(f"[INFO] Found {len(artist.songs)} songs. Saving to '{output_file}'...")

        with open(output_file, "w", encoding="utf-8") as f:          
            saved_count = 0
            for song in artist.songs:
                if song.lyrics:
                    f.write(song.lyrics)
                    saved_count += 1
                    
        print(f"[SUCCESS] Saved {saved_count} songs to '{output_file}'.")

    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

if __name__ == "__main__":
    fetch_and_save_lyrics(GENIUS_ACCESS_TOKEN, ARTIST_NAME, OUTPUT_FILENAME)
