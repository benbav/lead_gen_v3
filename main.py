from utils.scraper import TherapyLeadGenerator 
from utils.pinecone_client import PineconeVectorSearch
from utils.message_generator import MessageGenerator
import pandas as pd
import random
import time
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# TODO: clean this up later

api_key = os.getenv("PINECONE_API_KEY") 
BATCH_SIZE = 20  # seems to work ok

def main():
    print("Starting lead generation process...")
    
    # scrape potential leads from different sources

    # add linkedin in future version like below...
    # li_scraper = LinkedInScraper(keywords=["dashboard", "data visualization", "business intelligence"])
    therapy_scraper = TherapyLeadGenerator()  # new scraper for therapy business websites
    
    # Scrape therapy business websites (new addition)
    print("Scraping therapy business websites...")
    locations = ["new york", "los angeles", "chicago", "miami", "seattle"]
    therapy_results = therapy_scraper.generate_leads(locations=locations, max_results=30)
    print(f"Found {len(therapy_results)} therapy business leads")
    
    # convert therapy results to match other lead formats
    formatted_therapy_leads = []
    for lead in therapy_results:
        emails = lead["website_data"]["contact_emails"]
        email = emails[0] if emails else "unknown@example.com"
        
        formatted_therapy_leads.append({
            "name": lead["title"],
            "title": "Therapy Business Owner",
            "company": lead["title"],
            "platform": "Website",
            "profile_url": lead["url"],
            "recent_post": lead["website_data"]["about_content"][:100] + "...",
            "keywords_matched": lead["source_query"],
            "contact_info": email,
            "raw_website_data": lead["website_data"]  # keep the raw data for message generation
        })
    
    # combine results - prioritizing therapy leads since that's our new focus
    all_leads = formatted_therapy_leads
    random.shuffle(all_leads)  # mix them up a bit
    
    # save raw leads to csv
    leads_df = pd.DataFrame(all_leads)
    leads_df.to_csv(f"leads_{datetime.now().strftime('%Y%m%d')}.csv", index=False)
    
    # vectorize and find most promising leads
    pinecone = PineconeVectorSearch(api_key=os.environ.get("PINECONE_API_KEY", "pinecone_default_key"))
    pinecone.init_index("lead-matching-index")
    
    # our ideal client profile - now focusing on therapy businesses
    ideal_client = """Therapy business owner, established practice, using basic tools for tracking client progress, 
                    interested in data-driven decisions, has enough clients to need better visualization"""
    
    matched_leads = pinecone.match_leads(all_leads, ideal_client, top_k=BATCH_SIZE)
    
    # generate personalized messages
    msg_gen = MessageGenerator(model_path="./models/fine_tuned_llama2")
    
    messages = []
    for lead in matched_leads:
        try:
            if lead.get("platform") == "Website":
                # Use specific template for therapy businesses
                msg = msg_gen.generate_message(
                    lead_info=lead,
                    template="Hi there at {company}, I build dashboards for therapy practices to visualize client progress and practice metrics. Would you be interested in seeing how this could help {company}?"
                )
            else:
                msg = msg_gen.generate_message(
                    lead_info=lead,
                    template="Hi {name}, noticed your post about {keywords_matched}. Our dashboarding solution might help with your data visualization needs. Free demo?"
                )
            messages.append({"lead": lead, "message": msg})
        except Exception as e:
            print(f"Error generating message for {lead.get('name', 'unknown')}: {e}")
    
    # output results
    with open(f"outreach_messages_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", "w") as f:
        for item in messages:
            f.write(f"TO: {item['lead'].get('name', 'N/A')} ({item['lead'].get('platform', 'N/A')})\n")
            f.write(f"CONTACT: {item['lead'].get('contact_info', 'N/A')}\n")  # added contact info
            f.write(f"MESSAGE: {item['message']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"Generated {len(messages)} personalized outreach messages")
    print("Done!")

if __name__ == "__main__":
    main()

# lead generation pipeline for finding therapy business owners who might need dashboarding services
# scrapes leads from therapy business websites)
# uses pinecone vector search to match leads against an ideal client profile
# generates personalized outreach messages using a fine-tuned LLama2 model
# outputs a formatted list of potential clients with custom messages ready for outreach
# ADD EMAILING LATER