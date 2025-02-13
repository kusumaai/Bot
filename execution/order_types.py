@dataclass
class OrderDetails:
    # Required fields (no defaults)
    symbol: str
    side: str
    order_type: str
    size: Decimal
    price: Decimal
    
    # Optional fields (with defaults)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    time_in_force: str = 'GTC'
    post_only: bool = False
    reduce_only: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict) 