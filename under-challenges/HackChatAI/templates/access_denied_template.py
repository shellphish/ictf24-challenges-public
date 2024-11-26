access_denied_template = '''
<!doctype html>
<html>
    <head>
        <title>Access Denied</title>
        <!-- Bootstrap CSS -->
        <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        <style>
            body { 
                background-color: black; 
                color: #00ffea; 
                font-family: 'Courier New', Courier, monospace; 
                text-align: center;
                padding: 50px;
            }
            .container {
                margin-top: 50px;
                padding: 20px;
                border: 1px solid #00ffea;
                border-radius: 10px;
                background-color: #111;
                position: relative;
                overflow: hidden;
            }
            h2 {
                color: #ff00ff;
                position: relative;
                animation: glitch 1s infinite;
            }
            @keyframes glitch {
                0% {
                    transform: translate(0);
                }
                20% {
                    transform: translate(-2px, 2px);
                }
                40% {
                    transform: translate(-2px, -2px);
                }
                60% {
                    transform: translate(2px, 2px);
                }
                80% {
                    transform: translate(2px, -2px);
                }
                100% {
                    transform: translate(0);
                }
            }
            h2:before, h2:after {
                content: 'Access Denied';
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: #111;
                overflow: hidden;
            }
            h2:before {
                left: -2px;
                text-shadow: -1px 0 red;
                clip: rect(24px, 550px, 90px, 0);
                animation: glitchTop 1.5s infinite;
            }
            h2:after {
                left: 2px;
                text-shadow: -1px 0 blue;
                clip: rect(85px, 550px, 140px, 0);
                animation: glitchBottom 1.5s infinite;
            }
            @keyframes glitchTop {
                0% {
                    clip: rect(24px, 550px, 90px, 0);
                    transform: translate(-2px, -2px);
                }
                50% {
                    clip: rect(10px, 550px, 80px, 0);
                    transform: translate(2px, 2px);
                }
                100% {
                    clip: rect(24px, 550px, 90px, 0);
                    transform: translate(-2px, -2px);
                }
            }
            @keyframes glitchBottom {
                0% {
                    clip: rect(85px, 550px, 140px, 0);
                    transform: translate(2px, 2px);
                }
                50% {
                    clip: rect(80px, 550px, 100px, 0);
                    transform: translate(-2px, -2px);
                }
                100% {
                    clip: rect(85px, 550px, 140px, 0);
                    transform: translate(2px, 2px);
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Access Denied</h2>
        </div>

        <!-- Bootstrap JS and dependencies -->
        <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9CQZjlI6L/x3paNdF+JpG5Z1xM5rpbtI1kx78" crossorigin="anonymous"></script>
        <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
    </body>
</html>
'''
