import pandas as pd
from supabase import create_client, Client
import time
import os


# --- הגדרות (מלא את הפרטים שלך כאן) ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
INPUT_FILE = "reddit_data_ready_for_upload.csv"  #

# חיבור ל-Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_in_batches():
    print(f"Reading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
        total_rows = len(df)
        print(f"Total rows to upload: {total_rows}")

        # גודל כל מנה (Batch)
        batch_size = 1000

        # לולאה שרצה על הדאטה במנות
        for i in range(0, total_rows, batch_size):
            # חיתוך המנה הנוכחית
            batch = df.iloc[i: i + batch_size]

            # המרה לפורמט ש-Supabase מבין (רשימה של Dictionary)
            records = batch.to_dict(orient='records')

            try:
                # שליחה לטבלה 'knowledge_base'
                supabase.table('knowledge_base').insert(records).execute()

                # הדפסת התקדמות
                print(f"Uploaded rows {i} to {min(i + batch_size, total_rows)}...")

            except Exception as e:
                print(f"❌ Error uploading batch starting at index {i}: {e}")
                # אופציונלי: עצירה במקרה שגיאה, או פשוט להמשיך
                time.sleep(1)  # הפסקה קטנה במקרה של עומס

        print("✅ Upload process finished!")

    except FileNotFoundError:
        print(f"❌ Error: File '{INPUT_FILE}' not found.")


if __name__ == "__main__":
    upload_in_batches()