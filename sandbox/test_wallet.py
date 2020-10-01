import pytest
from wallet import Wallet, InsufficientAmount

@pytest.fixture
def my_wallet():
    '''Returns a Wallet instance with a zero balance'''
    return Wallet()

@pytest.mark.parametrize("earned,spent,expected", [
    (30, 10, 20),
    (20, 2, 18),
])

def test_transactions(my_wallet, earned, spent, expected):
    my_wallet.add_cash(earned)
    my_wallet.spend_cash(spent)
    assert my_wallet.balance == expected

def test_default_initial_amount(my_wallet):
    assert my_wallet.balance == 0

def test_setting_initial_amount(my_wallet):
    assert my_wallet.balance == 0

def test_wallet_add_cash(my_wallet):
    my_wallet.add_cash(80)
    assert my_wallet.balance == 80

def test_wallet_spend_cash(my_wallet):
    my_wallet.spend_cash(10)
    assert my_wallet.balance == 10

def test_wallet_spend_cash_raises_exception_on_insufficient_amount(my_wallet):
    with pytest.raises(InsufficientAmount):
        my_wallet.spend_cash(100)

