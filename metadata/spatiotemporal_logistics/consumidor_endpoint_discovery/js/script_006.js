jQuery(function($) {
	$('#acao-contraste a').click(function(event) {
		if ($.cookie('contraste') === null || $.cookie('contraste') != 'on') {
			$.removeCookie('contraste', {
				path : '/'
			});
			$.cookie('contraste', 'on', {
				path : '/'
			});
			$('body').addClass('contraste');
			event.preventDefault();
			return false;
		} else {
			$.removeCookie('contraste', {
				path : '/'
			});
			$.cookie('contraste', 'off', {
				path : '/'
			});
			$('body').removeClass('contraste');
			event.preventDefault();
			return false;
		}
	});

	if ($.cookie('contraste') == 'on') {
		$('body').addClass('contraste');
		return false;
	}
});
