from marshmallow import Schema, fields, validate

class PassengerSchema(Schema):
    Pclass = fields.Int(required=True, validate=validate.OneOf([1, 2, 3]))
    Name = fields.Str(required=True, validate=validate.Length(min=1))
    Sex = fields.Str(required=True, validate=validate.OneOf(["male", "female"]))
    Age = fields.Float(required=False)
    SibSp = fields.Int(required=True, validate=validate.Range(min=0))
    Parch = fields.Int(required=True, validate=validate.Range(min=0))
    Ticket = fields.Str(required=True)
    Fare = fields.Float(required=True, validate=validate.Range(min=0))
    Cabin = fields.Str(required=False)
    Embarked = fields.Str(required=False, validate=validate.OneOf(["C", "Q", "S"]))
