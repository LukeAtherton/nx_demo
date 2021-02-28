defmodule NxDemoTest do
  use ExUnit.Case
  doctest NxDemo

  test "greets the world" do
    assert NxDemo.hello() == :world
  end
end
