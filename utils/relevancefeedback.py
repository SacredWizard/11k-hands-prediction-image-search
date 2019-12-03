import os

from flask import Flask, request, render_template, send_from_directory, redirect

import task6_svm as task6_svm
import task6d_probability

app = Flask(__name__, template_folder=(os.path.abspath('templates')))
classifier_g = None
query_image_g = None
similar_images_g = None
port_g = 4558


def relevance_fdbk(data_dir, classifier, query_image, similar_images):
    """
    initial call to relevance feedback.
    Load app server
    Set global vars
    """
    print(data_dir)
    global classifier_g, query_image_g, similar_images_g

    classifier_g = classifier
    query_image_g = query_image
    similar_images_g = similar_images
    global port_g
    print("\nClick here: http://localhost:{0}/similar_images\n".format(port_g))
    app.run(port=port_g, debug=True)


@app.route(str('/Hands/<filename>'))
def send_image(filename):
    """
    Method to load Hand Images on browser page
    """
    return send_from_directory(os.path.abspath("Hands"), filename)


@app.route('/similar_images', methods=['GET', 'POST'])
def display_similar_images():
    """
    Method to handle feedback and return revised results
    """
    if request.method == 'POST':
        data = request.form
        global similar_images_g, query_image_g
        similar_images_g = incorporate_feedback(data.to_dict())
        # pdb.set_trace()
        return redirect("http://localhost:{0}/similar_images".format(port_g), code=303)
    elif request.method == 'GET':
        return render_template("relevancefeedback.html", image_names=[query_image_g, similar_images_g])


def incorporate_feedback(data):
    """
    Method to call the chosen classifier based feedback system
    """
    global classifier_g
    if classifier_g == "SVM":
        rel_similar_images = task6_svm.rerank_results(data)
    elif classifier_g == "PROBABILITY":
        rel_similar_images = task6d_probability.rerank_results(data)
    return rel_similar_images
