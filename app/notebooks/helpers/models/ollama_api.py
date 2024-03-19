import requests
import json


class OllamaAPI:
    """
    Python wrapper built around the Ollama API sever for inference
    Parameters:
    - model (str): Name of the model you want to use
    - endpoint (str, optional): Custom /generate endpoint for inference
    """

    def __init__(
        self,
        model,
        endpoint="http://172.29.208.1:11434/api/generate",
    ):
        self.model = model
        self.endpoint = endpoint

    def generate_response(
        self,
        prompt,
        images=None,
        return_format="json",
        options=None,
        system=None,
        template=None,
        context=None,
        stream=False,
        raw=False,
    ):
        """
        Generate a response using the specified parameters.
        Check Offical Docs for full examples: https://github.com/jmorganca/ollama/blob/main/docs/api.md

        Parameters:
        - prompt (str): The prompt to generate a response for (required).
        - images (list, optional): A list of base64-encoded images for multimodal models.
        - format (str, optional): The format to return a response in. Currently, the only accepted value is 'json'.
        - options (dict, optional): Additional model parameters listed in the documentation for the Modelfile.
        - system (str, optional): System message to override what is defined in the Modelfile.
        - template (str, optional): The prompt template to use, overrides what is defined in the Modelfile.
        - context (str, optional): The context parameter returned from a previous request to /generate,
                                    can be used to keep a short conversational memory.
        - stream (bool, optional): If True, the response will be returned as a a stream of objects,
                                    rather than single response object.
        - raw (bool, optional): If True, no formatting will be applied to the prompt.

        Returns:
        - dict: The response in JSON format.

        Example:
        >>> api = ModelAPI(model="llama2", endpoint="http://172.26.240.1:11434/api/generate")
        >>> response = api.generate_response(prompt="Why is the sky blue?")
        >>> print(response)
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "images": images,
            "format": return_format,
            "options": options,
            "system": system,
            "template": template,
            "context": context,
            "stream": stream,
            "raw": raw,
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(self.endpoint, data=json.dumps(data), headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()


if __name__ == "__main__":
    ollama = OllamaAPI("llama2")
    res = ollama.generate_response("What is your name?")
    print(res.get("response"))
