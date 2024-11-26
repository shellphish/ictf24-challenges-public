CREATE (:Person {firstName: "Stu", lastName: "Redman", tag: "Here you go, I lost my appetite all of a sudden."});
CREATE (:Person {firstName: "Abagail", lastName: "Freemantle", tag: "I have sinned in pride. So have you all. But that's past now."});
CREATE (:Person {firstName: "Frances", lastName: "Goldsmith", tag: "Harold, I don't think I'm ever gonna get these calluses off my fanny."});
CREATE (:Person {firstName: "Randall", lastName: "Flagg", tag: "ictf{People_who_try_hard_to_do_the_right_thing_always_seem_mad}"});
CREATE (:Person {firstName: "Larry", lastName: "Underwood", tag: "Baby, can you dig your man?"});
CREATE (:Person {firstName: "Harold", lastName: "Lauder", tag: "If you go with me, I’ll treat you like a queen. No, better than a queen. Like a goddess."});
CREATE (:Person {firstName: "Glen", lastName: "Bateman", tag: "The law is an imperfect mechanism. It doesn't operate at all when there is no enforcement."});

// Create friendships
MATCH (a:Person {firstName: "Stu"}), (b:Person {firstName: "Abagail"})
CREATE (a)-[:FRIEND]->(b);
CREATE (b)-[:FRIEND]->(a);

MATCH (a:Person {firstName: "Frances"}), (b:Person {firstName: "Abagail"})
CREATE (a)-[:FRIEND]->(b);
CREATE (b)-[:FRIEND]->(a);

MATCH (a:Person {firstName: "Stu"}), (b:Person {firstName: "Frances"})
CREATE (a)-[:FRIEND]->(b);
CREATE (b)-[:FRIEND]->(a);

MATCH (a:Person {firstName: "Frances"}), (b:Person {firstName: "Harold"})
CREATE (a)-[:FRIEND]->(b);
CREATE (b)-[:FRIEND]->(a);

MATCH (a:Person {firstName: "Larry"}), (b:Person {firstName: "Frances"})
CREATE (a)-[:FRIEND]->(b);
CREATE (b)-[:FRIEND]->(a);

MATCH (a:Person {firstName: "Glen"}), (b:Person {firstName: "Stu"})
CREATE (a)-[:FRIEND]->(b);
CREATE (b)-[:FRIEND]->(a);

MATCH (a:Person {firstName: "Larry"}), (b:Person {firstName: "Harold"})
CREATE (a)-[:FRIEND]->(b);
CREATE (b)-[:FRIEND]->(a);