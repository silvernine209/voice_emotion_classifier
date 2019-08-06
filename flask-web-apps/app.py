# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

from flask import Flask

app = Flask(__name__)  # create instance of Flask class


@app.route('/')  # the site to route to, index/main in this case
def hello_world() -> str:
    """Let's say Hi to the world.

    Returns:
        str: The HTML we want our browser to render.
    """

    return 'Hello World!'


if __name__ == '__main__':
    app.run()
