from flask import Flask,request, render_template, redirect, url_for
from flask import Blueprint

bp = Blueprint('main', __name__)

@bp.route('/')
def index():
    # will return json, but just setting up template to use in the future
    return 'Hello Word'