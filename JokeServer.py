from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/joke', methods=['GET'])
def PostJoke():
    try:
        joke = "This is a joke"
        print("Generated Joke:", joke)
        return jsonify({'fikra': joke})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='localhost', port=8080)
