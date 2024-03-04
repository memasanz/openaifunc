import azure.functions as func
from chatonyourdata import *
import logging

#3/4/2024 8:50 AM

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

def get_json_object(chat_history, question, content, citations):
    if chat_history is None:
        messages.append({"role": "user", "content": question})
        messages.append({"role": "system", "content": content})

    else:
        messages = chat_history
        messages.append({"role": "user", "content": question})
        messages.append({"role": "system", "content": content})
        messages.append({"role": "tool", "content": citations})

    
    data = {  
        'messages': messages,  
        'response': content
    }  
    json_object = json.dumps(data)
    return json_object

@app.route(route="http_trigger_chat_on_your_data")
def http_trigger_chat_on_your_data(req: func.HttpRequest) -> func.HttpResponse:
    try:
        logging.info('Python HTTP trigger function processed a request.')

        chat_history = []
        req_body = req.get_json()
        question = req_body.get('question')
        chat_history = req_body.get('chat_history')

        logging.info(f"Question: {question}")
        logging.info(f"Chat History Type: {type(chat_history)}")

        chatter = ChatOnYourData(index="good-fish", role="test-role")
        logging.info(f"Created ChatOnYourData object")
        content, citations = chatter.make_request(question, chat_history, includeCitationsInResponse=False)

        
        logging.info(f"response: {content}")
        

        return func.HttpResponse(get_json_object(chat_history, question, content, citations), status_code=200)
    except Exception as e:
        return func.HttpResponse(f"Error: {e}", status_code=500)
