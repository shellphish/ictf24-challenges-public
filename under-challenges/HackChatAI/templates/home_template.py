home_template = '''
<!doctype html>
<html>
    <head>
        <title>Home</title>
        <!-- Bootstrap CSS -->
        <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <style>
            body { 
                background-color: black; 
                color: #00ffea; 
                font-family: 'Courier New', Courier, monospace; 
            }
            .container {
                margin-top: 50px;
                padding: 20px;
                border: 1px solid #00ffea;
                border-radius: 10px;
                background-color: #111;
                text-align: center;
            }
            h2 {
                color: #ff00ff;
            }
            .btn-cyber {
                color: #00ffea;
                background-color: #111;
                border: 2px solid #00ffea;
                border-radius: 5px;
                padding: 10px 20px;
                font-size: 16px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                transition: all 0.3s ease;
            }
            .btn-cyber:hover {
                color: #111;
                background-color: #00ffea;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Welcome!</h2>
            <form action="/play" method="get">
                <button type="submit" class="btn-cyber">Play Game</button>
            </form>
        </div>

        <!-- Bootstrap JS and dependencies -->
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9CQZjlI6L/x3paNdF+JpG5Z1xM5rpbtI1kx78" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    </body>
</html>
'''
