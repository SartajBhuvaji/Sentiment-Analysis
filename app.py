from flask import Flask, render_template, request, redirect, url_for, flash
import os
from vader import Vader 
import bag_of_words.bag_of_words as bag_of_words
from draw_plot import draw_color_strip

app = Flask(__name__)
image_directory = os.path.join(os.getcwd(),'static')
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
MAX_RESULTS = os.getenv("MAX_RESULTS")

@app.route("/", methods=["POST", "GET"])
def home():
    if request.method == "POST":
        result = None
        video_url = request.form["video_url"]
        video_id = video_url.split("=")[1]
        url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={YOUTUBE_API_KEY}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults={MAX_RESULTS}'

        vader_obj = Vader()
        clened_comment, vader_result = vader_obj.runner(url)
        print(type(clened_comment))
        #print("CLEANEDD COMMENTS \n", clened_comment)
        bag_of_words_result = bag_of_words.runner(clened_comment)
        #result = f"Vader: {vader_result} \n Bag of Words: {bag_of_words_result}"
        #return render_template("index.html", result=result)

        draw_color_strip(vader_result, 'vader_result')
        draw_color_strip(bag_of_words_result, 'bag_of_words_result')

        vader_image_path = '/static/images/vader_result.png'
        bag_of_words_image_path = '/static/images/bag_of_words_result.png'
        return render_template("index.html", vader_image_path=vader_image_path, bag_of_words_image_path=bag_of_words_image_path)
    else:
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)