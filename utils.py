def get_trial_metadata(page_content, df):
    """
    Matches the retrieved document with the row in the DataFrame and extracts relevant fields.
    """
    for _, row in df.iterrows():
        concat = f"{row.get('Study Title','')} {row.get('Conditions','')} {row.get('Interventions','')} {row.get('Brief Summary','')}".strip()
        if page_content.strip()[:100] in concat[:200]:
            return {
                "Study Title": row.get('Study Title', ''),
                "NCT Number": row.get('NCT Number', ''),
                "Study Design": row.get('Study Design', ''),
                "Interventions": row.get('Interventions', ''),
                "Brief Summary": row.get('Brief Summary', ''),
            }
    return {}

def build_prompt(user_query, metadata):
    # Build a clean, minimal prompt to avoid repetition / redundant generation.
    # Only include essential trial info and the user question
    return f"""User Question:
{user_query}

Clinical Trial Information:
Title: {metadata.get('Study Title', 'Not Available')}
NCT Number: {metadata.get('NCT Number', 'Not Available')}
Study Design: {metadata.get('Study Design', 'Not Available')}
Interventions: {metadata.get('Interventions', 'Not Available')}
Brief Summary: {metadata.get('Brief Summary', 'Not Available')}
"""
