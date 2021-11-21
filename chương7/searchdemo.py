import cherrypy, os, urllib, pickle
import imagesearch
import random

class SearchDemo(object):

    def __init__(self) -> None:
        super().__init__()
        #load list of images
        with open('imname.pkl', 'rb') as f:
            self.imlist = pickle.load(f)
        
        self.nbr_images = len(self.imlist)
        self.ndx = range(self.nbr_images)

        # load vocabulary 
        with open('vocabulary.pkl', 'rb') as f:
            self.voc = pickle.load(f)
        
        # set max number of results to show
        self.maxres = 15

        # path to directory contain images 
        self.path = r"G:\dvtu\ThucTap\img\ukbench"

        # header ans footer html
        self.header = """
            <!doctype html>
            <head>
            <title>Image search example</title>
            </head>
            <body>
        """
        self.footer = """
            </body>
            </html>
        """
    
    def index(self, query=None):
        self.src = imagesearch.Searcher('test.db', self.voc)

        html = self.header
        html += """
            <br/>
                Click an image to search. <a href='?query='>Random selection</a> of images.
            <br/><br/>
        """
        if query:
            # query the database and get top images
            res = self.src.query(os.path.join(self.path, query))[:self.maxres]
            for dist, ndx in res:
                imname = self.src.get_filename(ndx)
                imname = os.path.basename(imname)
                html += "<a href='?query=" + imname + "'>"
                html += "<img src='" + imname + "'  width='100' />"
                html += "</a>"
        else:
            # show random selectioon if no query
            random.shuffle(list(self.ndx))
            for i in self.ndx[:self.maxres]:
                imname = self.imlist[i]
                imname = os.path.basename(imname)
                # imname = 'ukbench00003.jpg'
                html += "<a href='?query=" + imname + "'>"
                html += "<img src='" + imname + "' width='100' height='100' alt='Error link...' />"
                html += "</a>"
        html += self.footer
        return html

    index.exposed = True
cherrypy.quickstart(SearchDemo(), '/',
        config=os.path.join(os.path.dirname(__file__), 'service.conf'))
                


