import requests
from bs4 import BeautifulSoup
import time
import random
import json
import re
from fake_useragent import UserAgent
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

load_dotenv()  

class SearchAgent:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
        self.proxies = os.getenv("PROXY_LIST", "").split(",") if os.getenv("PROXY_LIST") else []
        self.current_proxy = 0
        self.req_count = 0
    
    def get_headers(self):
        return {"User-Agent": self.ua.random}
    
    def search(self, query, num=10):
        search_url = f"https://www.google.com/search?q={quote_plus(query)}&num={num}"
        
        # rotate proxy if we have any
        proxy = None
        if self.proxies:
            proxy = {"http": self.proxies[self.current_proxy], "https": self.proxies[self.current_proxy]}
            self.req_count += 1
            if self.req_count > 5:  # switch proxy every 5 requests
                self.current_proxy = (self.current_proxy + 1) % len(self.proxies)
                self.req_count = 0
        
        try:
            r = self.session.get(search_url, headers=self.get_headers(), proxies=proxy, timeout=10)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, 'html.parser')
            results = []
            
            # super brittle, will break if google changes layout
            for g in soup.select('div.g'):
                anchors = g.select('a')
                if not anchors:
                    continue
                
                link = anchors[0]['href']
                if link.startswith('/url?'):
                    link = link.split('&sa=')[0].replace('/url?q=', '')
                
                # skip certain domains
                if any(x in link for x in ['youtube.com', 'facebook.com', 'linkedin.com', 'instagram.com']):
                    continue
                    
                title_elem = g.select_one('h3')
                title = title_elem.text if title_elem else "No title"
                
                snippet_elem = g.select_one('div.VwiC3b')
                snippet = snippet_elem.text if snippet_elem else ""
                
                results.append({
                    "title": title,
                    "url": link,
                    "snippet": snippet
                })
                
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            time.sleep(5)  # backoff a bit
            return []


class WebsiteScraper:
    def __init__(self):
        self.ua = UserAgent()
        self.session = requests.Session()
    
    def get_headers(self):
        return {"User-Agent": self.ua.random}
    
    def scrape(self, url):
        try:
            r = self.session.get(url, headers=self.get_headers(), timeout=15)
            r.raise_for_status()
            
            soup = BeautifulSoup(r.text, 'html.parser')
            
            # extract contact info
            email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
            emails = re.findall(email_pattern, r.text)
            
            # Get content from common places
            about_content = ""
            about_sections = soup.select('.about, #about, [id*=about], [class*=about]')
            for section in about_sections:
                about_content += section.text.strip() + "\n"
            
            # Get meta description if available
            meta_desc = ""
            meta_tag = soup.find('meta', attrs={'name': 'description'})
            if meta_tag:
                meta_desc = meta_tag.get('content', '')
            
            # Extract some main content
            main_content = ""
            main_tags = soup.select('main, article, .content, #content')
            if main_tags:
                main_content = main_tags[0].text
            else:
                paragraphs = soup.select('p')
                for p in paragraphs[:10]:  # just get first 10 paragraphs
                    main_content += p.text + "\n"
            
            # Get services if they exist
            services = []
            service_sections = soup.select('.services, #services, [id*=service], [class*=service]')
            for section in service_sections:
                services.extend([li.text.strip() for li in section.select('li')])
            
            return {
                "url": url,
                "title": soup.title.text if soup.title else "",
                "contact_emails": list(set(emails)),
                "meta_description": meta_desc,
                "about_content": about_content[:500],  # truncate long content
                "main_content": main_content[:500],
                "services": services[:10]
            }
            
        except Exception as e:
            print(f"Scrape error for {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "title": "",
                "contact_emails": [],
                "about_content": "",
                "main_content": "",
                "services": []
            }


class TherapyLeadGenerator:
    def __init__(self):
        self.search_agent = SearchAgent()
        self.scraper = WebsiteScraper()
        
    def generate_leads(self, locations=["new york", "los angeles", "chicago"], max_results=20):
        all_leads = []
        
        for location in locations:
            # search for therapy businesses in different locations
            queries = [
                f"therapy practice {location}",
                f"counseling center {location}",
                f"mental health practice {location}"
            ]
            
            for query in queries:
                print(f"Searching for {query}...")
                search_results = self.search_agent.search(query)
                
                # limit results and avoid duplicates
                seen_domains = set()
                for result in search_results:
                    url = result["url"]
                    domain = url.split("//")[-1].split("/")[0]
                    
                    if domain in seen_domains:
                        continue
                    seen_domains.add(domain)
                    
                    print(f"Scraping {url}...")
                    website_data = self.scraper.scrape(url)
                    
                    # only add if we found at least an email
                    if website_data["contact_emails"]:
                        all_leads.append({
                            "title": result["title"],
                            "url": url,
                            "website_data": website_data,
                            "source_query": query
                        })
                    
                    if len(all_leads) >= max_results:
                        return all_leads
                    
                    # be nice to the websites
                    time.sleep(random.uniform(2, 5))
                
                # be nice to google too
                time.sleep(random.uniform(5, 10))
        
        return all_leads
    
    def generate_message(self, lead):
        website_data = lead["website_data"]
        
        # extract some useful info if available
        business_name = lead["title"].split("-")[0].strip()
        services = ", ".join(website_data["services"][:3]) if website_data["services"] else "therapy services"
        
        templates = [
            f"Hi there,\n\nI came across {business_name} while looking for therapy practices that might benefit from better data visualization. I build custom dashboards that help therapy businesses track client progress, appointment scheduling, and billing in one place.\n\nWould you be interested in seeing a demo?\n\nBest,\nAlex",
            
            f"Hello,\n\nI noticed {business_name} offers {services}. I've built dashboard tools specifically for therapy practices that make tracking client outcomes and practice metrics much easier.\n\nMany of my clients have been able to increase their client retention by 20% using these insights. Would you have 15 minutes this week to chat?\n\nCheers,\nAlex",
            
            f"Hi,\n\nI work with therapy practices like {business_name} to build custom data dashboards that simplify practice management. Would you be interested in a free consultation to see how this could work for your practice?\n\nThanks,\nAlex"
        ]
        
        return random.choice(templates)


if __name__ == "__main__":
    lead_gen = TherapyLeadGenerator()
    leads = lead_gen.generate_leads(max_results=5)
    
    # print(f"Found {len(leads)} leads with contact info")
    
    # Save results
    with open("therapy_leads.json", "w") as f:
        json.dump(leads, f, indent=2)
    
    # Generate sample messages
    for i, lead in enumerate(leads):
        message = lead_gen.generate_message(lead)
        print(f"\nSample message for {lead['title']}:")
        print(message)
        print("-" * 40)