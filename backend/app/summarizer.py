from transformers import pipeline

def summarize_text(text:str):
    summarizer = pipeline('summarization', model='philschmid/bart-large-cnn-samsum')
    summary = summarizer(text, max_length=150,min_length=50,do_sample=False)

    return summary[0]['summary_text']

if __name__ == '__main__':
    sample_text = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines. 
    It has become an essential part of the technology industry. Research associated with artificial intelligence 
    is highly technical and specialized. The core problems of artificial intelligence include programming computers 
    for certain traits such as knowledge, reasoning, problem-solving, perception, learning, planning, and ability to 
    manipulate and move objects.
    """
    print("Summary:", summarize_text(sample_text))
