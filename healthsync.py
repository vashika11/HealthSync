from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to HealthSync!"

# Add more routes below as needed
# Example:
# @app.route('/api/data', methods=['POST'])
# def receive_data():
#     data = request.get_json()
#     return jsonify({"status": "received", "data": data})

if __name__ == '__main__':
    app.run(debug=True)
