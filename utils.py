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
    return (
        f"User question: {user_query}\n\n"
        f"Using this clinical trial info:\n"
        f"Title: {metadata['Study Title']}\n"
        f"Study Design: {metadata['Study Design']}\n"
        f"Interventions: {metadata['Interventions']}\n"
        f"Brief Summary: {metadata['Brief Summary']}\n\n"
        "Provide a summary of the trial procedure relevant to the question."
    )
