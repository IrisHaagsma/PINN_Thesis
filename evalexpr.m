function [ varargout ] = evalexpr( varargin )
    if( ~nargin && ~nargout ), help evalexpr, return, end
    varargout = cell( 1, nargout );
    [varargout{:}] = featool( 'feval', 'evalexpr', varargin{:} );
    if( ~nargout ), clear varargout; 
end
