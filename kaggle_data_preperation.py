import pandas as pd

# שים כאן את הנתיב לקובץ שהורדת מ-Kaggle
input_file = r"C:\Users\User\Desktop\נושאים נבחרים\Project\Datasets\train_data_from_kaggle.csv"
output_file = "therapist_data_full_context.csv"

print(f"Reading {input_file}...")

try:
    df = pd.read_csv(input_file)

    # נוודא שהעמודות קיימות (בדרך כלל נקראות Context ו-Response)
    # ננרמל אותן לאותיות קטנות למקרה שיש הבדלים
    df.columns = df.columns.str.lower()

    if 'context' in df.columns and 'response' in df.columns:

        new_df = pd.DataFrame()

        # --- השינוי הגדול: איחוד שאלה ותשובה ---
        # אנחנו יוצרים פורמט ברור שהמודל יבין
        new_df['content'] = (
                "Patient Context: " + df['context'].fillna('') +
                "\n\nTherapist Response: " + df['response'].fillna('')
        )

        # נסמן את זה בנושא כללי כי אין לנו סיווג בקובץ הזה
        new_df['topic'] = 'general_therapy'

        # המקור
        new_df['origin'] = 'counsel_chat'

        # --- ניקוי ---
        # נסיר שורות שאין בהן תוכן אמיתי
        new_df = new_df[new_df['content'].str.len() > 30]

        # שמירה
        new_df.to_csv(output_file, index=False)

        print(f"✅ Success! Created '{output_file}'")
        print("Format example:")
        print(new_df.iloc[0]['content'][:100] + "...")  # נדפיס דוגמה שתראה
        print("-" * 30)
        print("Now upload this file to Supabase.")

    else:
        print("❌ Error: Could not find 'context' and 'response' columns.")
        print(f"Columns found: {df.columns.tolist()}")

except Exception as e:
    print(f"❌ Error: {e}")