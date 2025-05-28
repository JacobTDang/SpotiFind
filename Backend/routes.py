from flask import Flask,request, render_template, redirect, url_for
from app import app

def register_routes(app, db):
    # homepage
    app.route('/upload', methods=["GET", "POST"])
    def index():
        
        # will return json, but just setting up template to use in the future
        return ''