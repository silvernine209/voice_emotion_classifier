# What the Flask?

Flask is a popular python package that simplifies the process of building web apps.

# Demo in a tiny app

How does it work? Check out the tiny app in [app.py](app.py).

Let's look at the code here:
```python
# minimal example from:
# http://flask.pocoo.org/docs/quickstart/

from flask import Flask
app = Flask(__name__) # create instance of Flask class

@app.route('/') # the site to route to, index/main in this case
def hello_world():
    return 'Hello World!'

if __name__ == '__main__':
    app.run()

# runs as default if you $python app.py, instead of a module: http://thepythonguru.com/what-is-if-__name__-__main__/
```

The first two lines set up the base Flask framework. We'll make changes to the `app` object to make our webapp.
```python
from flask import Flask
app = Flask(__name__) # create instance of Flask class
```

The last two lines allow us to activate the web app whenever we run the file.
```python
if __name__ == '__main__':
    app.run()
```

Our web app is starting out really small. We are only building a single page on it, the index page. `@app.route('/')` is a decorator that tells the app to run the following function whenever anyone navigates to that page. The function returns the page we want to load.
```python
@app.route('/') # the site to route to, index/main in this case
def hello_world():
    return 'Hello World!'
```

Run the app by cd-ing into the directory where `app.py` lives. You'll start it out with `python3 app.py`.

You should see something like

```
* Serving Flask app "app" (lazy loading)
* Environment: production
  WARNING: Do not use the development server in a production environment.
  Use a production WSGI server instead.
* Debug mode: off
* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```

And then you can load up http://127.0.0.1:5000/ in your browser.

## Exercise: loading an image

**Step 1**

Replace that hello world page with a valid html page that loads an image of your choice. If you're new to html, refer to [this](https://www.w3schools.com/html/) to get you started.

**Step 2**

Now, save a variable in your app called `repeat_count` and have your app repeat the image a number of times equal to that count.

When you're finished, be sure to close out the app with ctl-c.

# Web app based on a model


## Step 1, saving a model


Check out the notebook where we store a model that predicts the species of flower based on attributes [here](train_save_model.ipynb)

## Step 2, Exercise: building an app that loads the model

Create a new app named `predictor_app.py`. Use our last app as a template. And make the following changes.

Add a new route like the one below.
```python
@app.route("/predict")
def predict():
    return "predict page placeholder"
```

This function is run for the URL http://127.0.0.1:5000/predict.

Now, add code above the routing function that loads the pickled model.

Verify that you're loading the pickled model by modifying the predict function to return `lr_model.feature_names`.

### Step 3, Exercise: build a web page that responds to app data

Flask uses a template engine to allow us to render webpages in response to the data in the app. Refer to the documentation for [Jinja2](http://jinja.pocoo.org/docs/2.10/) when you build your apps.


Say we want to build a list. Normally in HTML, we would do this like so:

```html
<ul>
  <li>Item 1</li>
  <li>Item 2</li>
  <li>Item 3</li>
</ul>
```

In Flask, we can use a template engine to build a list with code:

```html
<ul>
  {% for i in range(1,4) %}
      <li>Item {{ i }}</li>
  {% endfor %}
</ul>
```

Try this by creating an html file in the templates folder and inserting the flask template code. We can render templates by returning `return flask.render_template('predictor.html')`.

Verify that if you change the range, the list changes too.

Now, modify the above to list the feature names (hint: look at the [Flask Docs](http://flask.pocoo.org/docs/1.0/quickstart/#rendering-templates) for more info on passing arguments to a template).

### Step 3, Exercise: Forms

To do this step, we're going to first look at forms. Here's a form block that comes from w3schools.

```html
<form action="/action_page.php">
  First name:<br>
  <input type="text" name="firstname" value="Mickey">
  <br>
  Last name:<br>
  <input type="text" name="lastname" value="Mouse">
  <br><br>
  <input type="submit" value="Submit">
</form>
```

It has a few parts:
- The `<form action="/action_page.php"></form>` tag is a wrapper around the rest of the form. It includes the action attribute which tells us where to send the results of the form when we're through.
- Within that form are standard html things like text and `<br>`s for linebreaks.
- The `<input type="text" name="lastname" value="Mouse">` tag gives us an empty text field. We'll use this to grab user input. The `value` attribute sets the default value and `name` attribute tells us how to refer to the input block. Make sure that each `<input>` tag as a unique name. Also, note that `<input>`s are self-closing.
- The `<input type="submit" value="Submit">` tag is a special input that renders as a submit button.

Start with the form above and replace the input fields with our feature names:

```html
{% for f in feature_names %}
    <br>
    {{ f }}
    <br>
    <input type="text" name="{{ f }}" value="0">
{% endfor %}
```

Now change the form action so that it sends us back to the same page.

### Step 4, grab user input from a form and act on it.

Okay, so we're generating a form and that form sends us back to the same web page when we submit. Now, let's get a jump start on using a model by replacing your predict function with the following code

``` python
@app.route("/predict", methods=["POST", "GET"])
def predict():

    x_input = []
    for i in range(len(lr_model.feature_names)):
        f_value = float(
            request.args.get(lr_model.feature_names[i], "0")
            )
        x_input.append(f_value)

    pred_probs = lr_model.predict_proba([x_input]).flat

    return flask.render_template('predictor.html',
    feature_names=lr_model.feature_names,
    x_input=x_input,
    prediction=list(np.argsort(pred_probs)[::-1])
    )
```

Now, update your template to include

```html
<form action="/predict">

  {% for f in feature_names %}
      <br>
      {{ f }}
      <br>
      <input type="text" name="{{ f }}" value="{{x_input[loop.index0]}}">
  {% endfor %}
  <br>
  <input type="submit" value="Submit" method="get">



</form>

<p>
  {% if prediction %}
    prediction: {{ prediction|safe}}
  {% endif%}
</p>
```



How do we get the user input? Where's it stored?

Fire up the app with `python3 predictor_app.py`. Go to the predict page [http://localhost:5000/predict](http://localhost:5000/predict)

What do you notice changes when you click submit?

Check out the url.

This is one way that web apps communicate, both internally and between API calls. We can grab those values by using `request.args`, which acts as a dictionary. It's good practice to use `request.args.get(key, default)` so that your app can gracefully use a default value when no key is available.


### Further Exercises

- Modify the app to output the predicted class (so 'setosa' instead of [0]).
- Modify the app to output the classes with their predicted probabilities. Sort them.
- This is a somewhat ugly app, can you make it prettier?

When you're finished, check out the [cancer app](cancer_app) for a preview of what we can do withe d3.
