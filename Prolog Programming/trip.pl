

/**
* Shubhi Nigam - SBU ID 112672816
* Rishabh Khot - SBU ID 112682983
*/


/**
* check_op finds all the members in the operator list to get the total count of each operator and validating if it is below or equal the required condition.
* Here findall function gets all the operator list and validates with the maxacceptedoffer, there is an additional check that needs to be done with the length that is been handled.
*/
check_op([], _).
check_op([O|T], Sub) :-
	findall(P, member((O,P,_LA), Sub), L),
	maxacceptedoffer(M),
	length(L,N),
	N =< M,
	check_op(T, Sub).


/**
* the length function calculates the length of the list
*/

length([],0).
length([_|T],N):-
	length(T,N1),
	N is N1+1.


/**
* satisy_need validates if for each activity the needs are getting satisfied or not
*/

satisfy_need(_,U,_):-
	U =< 0,
	!.
	
satisfy_need(A,U, [(_O,_P,LA)|T]):-
	member((A, U2), LA)
	-> NU is U - U2, satisfy_need(A, NU, T)
	; satisfy_need(A, U, T).


/**
* member function creates a group of operator and activity
*/

member(H, [H|_]).
member(X, [_|T]) :-
	member(X, T).


/**
* here we get all the packages combinations bu using findall
* findall function gets all the 
*/

collect_packages([], []).
collect_packages([(O,P)|LOP], [(O,P,LAU)|LOPAU]):-
	findall((A,U),offer(O,P,A,U),LAU),
	collect_packages(LOP, LOPAU).




needs(LNeeds) :-
	findall((A,U), need(A,U), LNeeds).


collect_all_offers(LOP_AU) :-
	setof((O,P), X^Y^offer(O, P, X, Y), LOP),
	collect_packages(LOP, LOP_AU).


subset([],[]).
subset([H|T],[H|T2]):-
	subset(T,T2).
subset([_|T],T2):-
	subset(T,T2).


/**
* here we check the condition of operator is getting satisfied or not
*/
check_max_per_op(Sub) :-
	setof(O, P^LA^member((O,P,LA), Sub), LO),
	check_op(LO, Sub).


iterate_needs([], _).
iterate_needs([(A,U)|T],Sub):-
	satisfy_need(A,U,Sub),
	iterate_needs(T, Sub).



/**
* calculting the price based on the final list of operator and activity 
*/

compute_price([],0).
compute_price([(O, P, _)|Sub], Price) :-
	price(O,P,Price1),
	compute_price(Sub, Price2),
	Price is Price1 + Price2.


/**
* The subset created here collects all the offers and creates subsets of the offers, after that we check if the max operator conditio  has reached or not and finally we compute the price of the offer and the activity
*/
ok_subset(Sub, Price) :-
	needs(LNeeds),
	collect_all_offers(LOP_AU),
	subset(LOP_AU, Sub),
	check_max_per_op(Sub),
	iterate_needs(LNeeds, Sub),
	compute_price(Sub, Price).



/**
*   The min list function here validates the condition of the needs met by the sub list or not
*/
min_list([H], H) :-
	!.
	
min_list([H|T], M) :-
	min_list(T, M2),
	(H<M2 -> M=H; M=M2).




/**
* calling the totalcost function for the computation of the cost of the operator and activity
*/
totalcost(M) :-
	findall(Price, ok_subset(_Sub, Price), LPrices),
	min_list(LPrices, M).

