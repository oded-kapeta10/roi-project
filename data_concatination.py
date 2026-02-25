import os
import pandas as pd

# --- Configuration ---
root_folder_path = r"C:\Users\User\Desktop\נושאים נבחרים\Project\Datasets\Original Reddit Data"

# --- UPDATED TOPIC MAP ---
topic_map = {
    # Anxiety
    "anx": "anxiety",
    "ani": "anxiety",  # Added for files like 'anijun19'

    # Depression
    "dep": "depression",

    # Loneliness
    "lone": "loneliness",
    "lon": "loneliness",  # Added for files like 'lonmar19'

    # Suicide Watch
    "sui": "suicide_watch",
    "sw": "suicide_watch",

    # Mental Health
    "mental": "mental_health",
    "mh": "mental_health",  # Added for files like 'mhaug19'

    # Labelled Data (LD)
    "ld ": "labelled_data",  # רווח אחרי LD כדי לא לתפוס מילים אקראיות
    "ld_": "labelled_data"
}

all_data = []

print("Starting to scan folders and merge data...")

for root, dirs, files in os.walk(root_folder_path):
    for file in files:
        if file.endswith(".csv"):
            file_path = os.path.join(root, file)
            filename_lower = file.lower()

            # Identify topic
            current_topic = "general"
            for key, value in topic_map.items():
                if key in filename_lower:
                    current_topic = value
                    break

            # אם זה קובץ Labelled Data ספציפי (לפי הלוגים שלך)
            if file.startswith("LD "):
                current_topic = "labelled_data"

            try:
                # Read CSV
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except:
                    df = pd.read_csv(file_path, encoding='latin1')

                # Combine text
                if 'title' in df.columns and 'selftext' in df.columns:
                    df['content'] = df['title'].fillna('') + " " + df['selftext'].fillna('')
                elif 'text' in df.columns:
                    df['content'] = df['text'].fillna('')

                # Process valid content
                if 'content' in df.columns:
                    temp_df = df[['content']].copy()
                    temp_df['topic'] = current_topic
                    temp_df['origin'] = 'reddit'

                    # Filter bad rows
                    temp_df = temp_df[~temp_df['content'].str.contains('\[deleted\]|\[removed\]', na=False)]
                    temp_df = temp_df[temp_df['content'].str.len() > 20]

                    all_data.append(temp_df)

            except Exception as e:
                print(f"Skipping {file}: {e}")

# Save final result
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    output_filename = "final_reddit_data_for_supabase.csv"
    final_df.to_csv(output_filename, index=False)

    print(f"✅ Success! Created file: {output_filename}")
    print(f"Total rows: {len(final_df)}")

    # בדיקה סופית: האם נשארו קבצי general?
    general_count = len(final_df[final_df['topic'] == 'general'])
    print(f"Rows marked as 'general': {general_count}")

else:
    print("❌ No valid data found.")