from flask import Flask, request, jsonify
from gpt2.gpt2_model import generate_gpt2_response

def create_app():
    app = Flask(__name__)

    @app.route('/message', methods=['POST'])
    def process_message():
        data = request.get_json()
        message = data.get('message', '')

        # Generar respuesta usando GPT-2
        responses = generate_gpt2_response(message)
        response = responses[0]  # Tomar la primera respuesta generada

        return jsonify({"response": response})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="localhost", port=5000)