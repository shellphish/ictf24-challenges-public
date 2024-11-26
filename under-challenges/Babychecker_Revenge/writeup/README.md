# Babychecker_Revenge



This is the revenge version Baby SSRF Checker. In this challenge, we still need to send a request to http://127.0.0.1:5000/flag. but there's a limitation of 50 words for the input.So it's hard to prompt injection. But this time, we can noticed that that there isn't an `allow_redirect=False` in `requests.get`, so we can solve the challenge with  `302 redirect`.



Firstly, we hold a server on our own server, an example is like:

```php
<?php
header("Location: http://127.0.0.1:5000/flag");
?>
```

then we input our link, the server will visit our webisit, and redirect to the flag page.





