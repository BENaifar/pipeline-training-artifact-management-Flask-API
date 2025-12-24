from flask import Flask
from app.routes.passenger_routes import bp as passenger_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(passenger_bp)
    return app
