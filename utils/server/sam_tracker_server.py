from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/receive_position', methods=['POST'])
def receive_position():
    data = request.get_json()
    
    # Extract x, y, z from the received data
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')

    if x is None or y is None or z is None:
        return jsonify({"error": "Invalid data received"}), 400

    print(f"Received position - X: {x}, Y: {y}, Z: {z}")

    return jsonify({"message": "Position received successfully"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
