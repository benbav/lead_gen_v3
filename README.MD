# Lead Generator for Therapy Business Owners

Quick tool i put together to generate leads for my dashboarding business. Targets therapy business owners who need better data viz.

## What it does

- Scrapes therapy business owner websites using `utils/scraper.py`
- Generates personalized outreach messages with `utils/message_generator.py`
- Stores data in Pinecone vector db for easy retrieval via `utils/pinecone_client.py`
- Uses fine-tuned LLama2 to make messages sound more human (see `fine_tune_llama2.py`)
 - Implemented LoRA (Low-Rank Adaptation) since I was running on a Mac without GPUs
 - Still got decent results even without serious hardware - but might train on cloud with free credits later
- want to add email capabilities in the future but still messing with GCP to see how easy that would be 

## Getting Started

```
pip install -r requirements.txt
python main.py
```
You can also check out `test.py` for examples but its still a work in progress

## Todo

- Fix bug in scraper that sometimes misses contact info
- Add more templates to message generator
- Improve the fine-tuning process (currently overfits on small dataset)
- Add logging

## Notes

I made this fast so there's probably still bugs. It's worked well enough to get me 15 new clients last month though.

This is just the base version - I'm working on turning this into a full AI agent that can handle the whole outreach process without my input.