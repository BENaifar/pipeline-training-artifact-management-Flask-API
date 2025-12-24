from flask import Blueprint, request, render_template, jsonify
from app.schemas.passenger_schema import PassengerSchema
from app.services.predict_service import predict_passenger

bp = Blueprint('passenger', __name__)
schema = PassengerSchema()

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template("home.html")
    
    form_data = request.form.to_dict()
    errors = schema.validate(form_data)
    
    if errors:
        # Return form data + errors to template
        return render_template("home.html", errors=errors, form_data=form_data)
    
    # Call service
    result = predict_passenger(form_data)
    return render_template("home.html", results=result, form_data=form_data)

