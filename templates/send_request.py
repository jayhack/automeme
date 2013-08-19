import requests

def send_request (top_text, bottom_text, meme_type):

	payload = {	'generatorID': 305, 
				'imageID': 84688,
				'text0': 'hello',
				'text1': 'world',
				}

	r = requests.post(r'www.URL.com/home',data=payload)

    print r.text