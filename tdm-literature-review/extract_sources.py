from pathlib import Path
import re, html

p = Path(__file__).parent / 'sources'
for f in p.glob('2609*.html'):
    text = html.unescape(re.sub('<[^>]+>', ' ', f.read_text(encoding='utf-8')))
    text = re.sub(r'[ \t]+', ' ', text)
    f.with_suffix('.txt').write_text(text, encoding='utf-8')
s = (p / 'cfpb-archive.html').read_text(encoding='utf-8')
for u in re.findall(r'href=[\"\x27]([^\"\x27]+)', s):
    if 'files.consumerfinance' in u:
        print(u)
