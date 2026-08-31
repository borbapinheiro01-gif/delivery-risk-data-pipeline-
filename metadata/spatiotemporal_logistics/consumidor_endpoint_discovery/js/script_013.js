/**
 * Componente customizado usando jQueryUI que adiciona o evento onChange no combobox atualizando um segundo combobox pela url passada para o componente.
 * 
 */
(function() {
    // the widget definition, where "custom" is the namespace,
    // "colorize" the widget name
    $.widget("custom.changeCombobox", {
        // default options
        options: {
            updateCombobox: "",
            updateUrl: "",
            defaultItemText: "Selecione",
            onupdate: null,
            onposupdate: null,
            buscarnull: false,
            requestParam:"",
	    changeDisabled:true
        },
        getSelectedValue: function() {
            return this.element.val();
        },
        // the constructor
        _create: function() {
            // cria o evento onChange do componente
            this._on(this.element, {
                change: "onchange"
            });
        },
        onchange: function() {
            var updateCombo = this.options.updateCombobox;
            var updateUrl = this.options.updateUrl;
            var buscarnull = this.options.buscarnull;
            var requestParam = this.options.requestParam;

            if (updateCombo && updateUrl) {
                var id = this.getSelectedValue();// obtem o valor do combo dependente
                var defaultItemText = this.options.defaultItemText;
                var $selectFilha = $(updateCombo);

                $selectFilha.html('');
                $selectFilha.append('<option value="">' + defaultItemText + '</option>');

                var $instance = this;

                if (id && !buscarnull) {
                    if (updateUrl.indexOf("$id") < 0) {
                        updateUrl = updateUrl + '/' + id + '.json';
                    } else {
                        updateUrl = updateUrl.replace("$id", id);
                    }
                    
                    if(requestParam != "") {
                    	updateUrl = updateUrl+"?"+requestParam;
                    }
                    
                    $.getJSON(updateUrl, function(data) {
                        updateUrl = updateUrl + '/' + "null" + '.json';
                        $instance._onupdate($selectFilha, data);
                    }).fail(function( xhr, textStatus, error ) {
                        exibirMensagemAlertaOuAjax(xhr.status, xhr.responseText);
                    });
                }

                else if (buscarnull) {

                    if (!id) {
                        id = " ";
                    }

                    if (updateUrl.indexOf("$id") < 0) {
                        updateUrl = updateUrl + '/' + id + '.json';
                    } else {
                        updateUrl = updateUrl.replace("$id", id);
                    }
                    $.getJSON(updateUrl, function(data) {
                        $instance._onupdate($selectFilha, data);
                    }).fail(function( xhr, textStatus, error ) {
                        exibirMensagemAlertaOuAjax(xhr.status, xhr.responseText);
                    });
                }
                else {
			if(this.options.changeDisabled) {
				$selectFilha.attr('disabled', true);
			}
                    $selectFilha.change();
                }
            }
        },
        _onupdate: function(elemento, data) {

            if (this.options.onupdate) {
                this.options.onupdate(this, elemento, data);
            } else {
                $.each(data, function(key, val) {
                    elemento.append('<option value="' + val.value + '">' + val.label + '</option>');
                });
            }

            // desabilita se estiver vazio
	    if(this.options.changeDisabled) {
		var disabled = (data.length && data.length == 0);
		elemento.attr('disabled', disabled);
	    }

            if (this.options.onposupdate) {
                this.options.onposupdate();
            }
        },
        // events bound via _on are removed automatically
        // revert other modifications here
        _destroy: function() {
            // remove generated elements
        },
        // _setOptions is called with a hash of all options that are changing
        // always refresh when changing options
        _setOptions: function() {
            // _super and _superApply handle keeping the right this-context
            this._superApply(arguments);
        },
        // _setOption is called for each individual option that is changing
        _setOption: function(key, value) {
            this._super(key, value);
        }
    });
})(jQuery);