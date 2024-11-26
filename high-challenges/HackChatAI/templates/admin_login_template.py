# Admin login page template
admin_login_template = '''
<!doctype html>
<html>
    <head>
        <title>Admin Login</title>
    </head>
    <body>
        <h2>Admin Login</h2>
        <form action="/admin-login" method="post">
            Cookie: <input type="text" name="cookie"><br>
            <input type="submit" value="Login">
        </form>
        {% if error %}
            <p style="color: red;">{{ error }}</p>
        {% endif %}
    </body>
</html>
'''
